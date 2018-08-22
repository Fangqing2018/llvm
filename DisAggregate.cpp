//===- DisAggregate.cpp - Decompose structures -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass decompose structures when necessary. For eg: when the struct
// contains pointers
// 1. Then there is bitcast on the struct, then don't do the disaggregation
// 2. This pass will not dis-aggregate the recursive type
//
//===----------------------------------------------------------------------===//

#include "reflow/LinkAllPasses.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CallSite.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Use.h"
#include "llvm/IR/User.h"
#include "llvm/IR/Value.h"
#include "llvm/IR/GetElementPtrTypeIterator.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Analysis/CallGraph.h"
#include <cassert>
#include <cstdint>
#include <utility>
#include <vector>

using namespace llvm;

#define DEBUG_TYPE "disaggr"


namespace {

/// DisAggregate - The structure decomposition pass.
class DisAggregate: public ModulePass,
                    public InstVisitor<DisAggregate, bool> {

  // then InstVisitor can access private members
  friend class InstVisitor<DisAggregate, bool>;
  std::unique_ptr<CallGraph> CG;
  DenseMap<Value*, SmallVector<Value*, 5>> mReplacementMap;
  SetVector<PHINode*> PHINodeSet;

public:
  static char ID; // Pass identification, replacement for typeid 
  DisAggregate() : ModulePass(ID) {
    initializeDisAggregatePass(*PassRegistry::getPassRegistry());
  }

  bool runOnModule(Module &M) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<CallGraphWrapperPass>();
  }

private:
  bool runOnCallGraphPostOrder();
  bool runOnFunction(Function *F);
  void addReplacementValue(Value* OrigVal, Value* NewVal);
  bool hasReplacement(Value* Val);
  void getReplacement(Value* Val, SmallVectorImpl<Value*>& Repl,
                                  bool recursive = false);
  Function *visitFunctionArgs(Function *F);
  bool visitConstantData(Constant *C);
  bool visitGlobalVariables(Module &M);
  bool visitGlobalVariable(GlobalVariable *G);
  bool visitAllocaInst(AllocaInst &AI);
  bool visitLoadInst(LoadInst &LI);
  bool visitStoreInst(StoreInst &SI);
  bool visitGetElementPtrInst(GetElementPtrInst &GEP);
  bool visitCallInst(CallInst& CI);
  bool visitExtractValueInst(ExtractValueInst &EVI);
  bool visitInsertValueInst(InsertValueInst &IVI);
  bool visitPHINode(PHINode &PN);
  bool visitPHINodesAgain();
  bool visitPHINodeAgain(PHINode *PN);
  bool visitSelectInst(SelectInst &SI);
  bool visitInstruction(Instruction &I);

  bool eraseInstructions(Instruction *I);
};

} // end anonymous namespace

char DisAggregate::ID = 0;

INITIALIZE_PASS_BEGIN(DisAggregate, DEBUG_TYPE, "Structure Decomposition",
                      false, false)
INITIALIZE_PASS_DEPENDENCY(CallGraphWrapperPass)
INITIALIZE_PASS_END(DisAggregate, DEBUG_TYPE, "Structure Decomposition",
                      false, false)

static bool needToBeDecomposed(Type* Ty) {
  assert(Ty != 0);

  // If is primitive type, then return
  if (Ty->isSingleValueType() && !Ty->isPointerTy())
    return false;

  // Strip off the sequential types.
  while (Ty->isArrayTy() || Ty->isVectorTy() || Ty->isPointerTy()) {
    if (Ty->isArrayTy() || Ty->isVectorTy())
      Ty = Ty->getSequentialElementType();
    else
      Ty = Ty->getPointerElementType();
  }

  auto *STy = dyn_cast<StructType>(Ty);

  // Not a struct
  if (!STy)
    return needToBeDecomposed(Ty);

  // Check each element of struct
  for (unsigned i=0; i<STy->getNumElements(); i++) {
    auto *ElementTy = STy->getElementType(i);
    if (ElementTy->isPointerTy()) {
      // Check recursive type
      // FIXME: maybe we need move this into a standalone checker.
      if (ElementTy->getPointerElementType() == STy) {
        dbgs() << "Recursive type\n";
        return false;
      }
      return true;
    } else {
      if (needToBeDecomposed(ElementTy))
        return true;
    }
  }

  return false;
}

static bool needToBeDecomposed(Value* Val) {
  return needToBeDecomposed(Val->getType());
}

static StructType* getFirstStructType(Type *Ty) {
  // Strip off the sequential types.
  while (Ty->isArrayTy() || Ty->isVectorTy() || Ty->isPointerTy()) {
    if (Ty->isArrayTy() || Ty->isVectorTy())
      Ty = Ty->getSequentialElementType();
    else
      Ty = Ty->getPointerElementType();
  }

  return cast<StructType>(Ty);
}

// Peel one layer at once
static void getDecomposedStructTypes(Type* Ty, SmallVectorImpl<Type*>& NewTypes) {
  assert(needToBeDecomposed(Ty));

  // Strip off the sequential types.
  SmallVector<Type*, 3> SeqTys;
  while (Ty->isArrayTy() || Ty->isVectorTy() || Ty->isPointerTy()) {
    SeqTys.push_back(Ty);
    if (Ty->isArrayTy() || Ty->isVectorTy())
      Ty = Ty->getSequentialElementType();
    else
      Ty = Ty->getPointerElementType();
  }

  // sub types.
  SmallVector<Type*, 3> SubTys;
  // Handle regular struct type.
  StructType* STy = cast<StructType>(Ty);

  for (unsigned i = 0; i < STy->getNumElements(); ++i)
    SubTys.push_back(STy->getElementType(i));

  // Add back the wrapping sequential types.
  for (unsigned i = 0; i < SubTys.size(); ++i) {
    Type* SubTy = SubTys[i];
    for (unsigned k = SeqTys.size(); k > 0; --k) {
      Type* SeqTy = SeqTys[k-1];
      
      if (PointerType* PTy = dyn_cast<PointerType>(SeqTy))
        SubTy = PointerType::get(SubTy, PTy->getAddressSpace());
      else if (ArrayType* ATy = dyn_cast<ArrayType>(SeqTy))
        SubTy = ArrayType::get(SubTy, ATy->getNumElements());
      else if (VectorType* VTy = dyn_cast<VectorType>(SeqTy))
        SubTy = VectorType::get(SubTy, VTy->getNumElements());
      else
        assert(0 && "Other sequential types?"); // Attn: Do not allow a packed type.
    }

    NewTypes.push_back(SubTy);
  }
}

#if 0
static bool getDecomposedFunctoinType(FunctionType* FTy, FunctionType* NewFTy) {
  bool Changed = false;
  // sub types.
  SmallVector<Type*, 3> SubTys;
  // Handle function type.
  SmallVector<Type*, 5> NewParamTys;

  assert(!FTy->isVarArg() && "Don't support function with variable args");
  for (unsigned i = 0; i < FTy->getNumParams(); ++i) {
    Type* ParamTy = FTy->getParamType(i);
    if (needToBeDecomposed(ParamTy)) {
      SmallVector<Type*, 3> OneParamTys;
      getDecomposedStructTypes(ParamTy, OneParamTys);
      NewParamTys.insert(NewParamTys.end(), OneParamTys.begin(),
                         OneParamTys.end());
      Changed = true;
    } else
      NewParamTys.push_back(ParamTy);
  }
  if (!Changed) {
    NewFTy = nullptr;
    return false;
  }

  NewFTy = FunctionType::get(FTy->getReturnType(),
                                           NewParamTys,
                                           FTy->isVarArg());

  return true;
}
#endif

static void setStructFieldVarName(Value* NewVal, Value* OldVal, unsigned idx) {
    assert(NewVal && OldVal && "");
    
    std::string NewName = OldVal->getName();
    NewName += "." + Twine(idx).str();

    NewVal->setName(NewName);
}

bool DisAggregate::eraseInstructions(Instruction *I) {
  if (!I->use_empty())
    return false;

  // Collect operands
  SmallVector<Instruction*, 3> Ops;
  for (Use &Op: I->operands()) {
    auto *Inst = dyn_cast<Instruction>(Op.get());
    if (Inst) {
      if (isa<PHINode>(Inst))
        continue;
      Ops.push_back(Inst);
    }
  }

  //Function* F = nullptr;
  //if (auto *CI = dyn_cast<CallInst>(I)) {
  //  F = CI->getCalledFunction();
  //}

  I->eraseFromParent();

  //if (F && F->use_empty()) {
  //  auto *CGNode = (*CG)[F];
  //  if (CGNode->getNumReferences() == 0)
  //    delete CG->removeFunctionFromModule(CGNode);
  //  else
  //    F->setLinkage(Function::ExternalLinkage);
  //}

  for (auto *Op: Ops)
    eraseInstructions(Op);

  return true;
}

bool DisAggregate::runOnModule(Module &M) {
  bool Changed = false;


  bool LocalChanged = false;
  do {
    LocalChanged = false;
    CG.reset(new CallGraph(M));
    // runOnGlobalVariables();
    LocalChanged |= runOnCallGraphPostOrder();
    Changed |= LocalChanged;
  } while(LocalChanged);

  return Changed;
}
bool DisAggregate::runOnCallGraphPostOrder() {
  bool Changed = false;

  Changed |= visitGlobalVariables(CG->getModule());

  auto *Root = CG->getExternalCallingNode();
  std::set<CallGraphNode*> Visited;
  // Use vector instead of traverse the post_order, because the CallGraph may be
  // updated during the process.
  std::vector<CallGraphNode*> PostVec(po_ext_begin(Root, Visited),
                                      po_ext_end(Root, Visited));
  for (auto *N: PostVec) {
    Changed |= runOnFunction(N->getFunction());
  }

  return Changed;
}

bool DisAggregate::visitGlobalVariables(Module &M) {
  bool Changed = false;

  SmallVector<GlobalVariable*, 16> GlobalVals;
  for (auto &GI: M.globals())
    GlobalVals.push_back(&GI);

  for (auto *GI: GlobalVals)
    Changed |= visitGlobalVariable(GI);
  return Changed;
}

bool DisAggregate::visitGlobalVariable(GlobalVariable *G) {
  auto *Ty = G->getType();
  if (!needToBeDecomposed(Ty))
    return false;
  if (hasReplacement(G))
    return false;

  SmallVector<Value*, 5> Initials;
  if (G->hasInitializer()) {
    visitConstantData(G->getInitializer());
    getReplacement(G->getInitializer(), Initials);
  }

  SmallVector<Type*, 5> NewTys;
  getDecomposedStructTypes(cast<PointerType>(Ty)->getElementType(), NewTys);
  unsigned NumElements = NewTys.size();
  assert(NumElements && "No member in struct?");
  assert(NumElements == Initials.size() && "mismatch?");
 
  for (unsigned i = 0; i < NumElements; i++) {
    GlobalVariable* NewG = new GlobalVariable(*G->getParent(),
                                              NewTys[i],
                                              G->isConstant(),
                                              G->getLinkage(),
                                              G->hasInitializer() ?
                                                cast<Constant>(Initials[i]):
                                                nullptr,
                                              "",
                                              G);
    setStructFieldVarName(NewG, G, i);
    addReplacementValue(G, NewG);
  }

  return true;
}

static Constant* visitConstantDataByIndex(Constant *C, unsigned Idx) {
  SmallVector<Type*, 5> NewTys;
  getDecomposedStructTypes(C->getType(), NewTys);
  
  if (C->isNullValue())
    return Constant::getNullValue(NewTys[Idx]);

  if (isa<UndefValue>(C))
    return UndefValue::get(NewTys[Idx]);

  if (auto *ConstStruct = dyn_cast<ConstantStruct>(C))
    return ConstStruct->getOperand(Idx);

  if (auto *ConstDataArray = dyn_cast<ConstantDataArray>(C)) {
    SmallVector<Constant*, 5> Elements;
    for (unsigned i = 0; i < ConstDataArray->getNumElements(); i++)
      Elements.push_back(
        visitConstantDataByIndex(ConstDataArray->getElementAsConstant(i), Idx));

    return ConstantArray::get(cast<ArrayType>(NewTys[Idx]), Elements);
  }

  if (auto *ConstArray = dyn_cast<ConstantArray>(C)) {
    SmallVector<Constant*, 5> Elements;
    for (unsigned i = 0; i < ConstArray->getNumOperands(); i++)
      Elements.push_back(
        visitConstantDataByIndex(ConstArray->getOperand(i), Idx));

    return ConstantArray::get(cast<ArrayType>(NewTys[Idx]), Elements);
  }

  assert(0 && "Other types");
  return nullptr;
}

bool DisAggregate::visitConstantData(Constant *C) {
  auto *Ty = C->getType();
  if (!needToBeDecomposed(Ty))
    return false;
  if (hasReplacement(C))
    return false;

  SmallVector<Type*, 5> NewTys;
  getDecomposedStructTypes(Ty, NewTys);
  unsigned NumElements = NewTys.size();

  for (unsigned i = 0; i < NumElements; i++) {
    auto * NewC = visitConstantDataByIndex(C, i);
    addReplacementValue(C, NewC);
  }

  return true;
}

bool DisAggregate::runOnFunction(Function *F) {
  if (!F || (F->isDeclaration() && F->use_empty()))
    return false;

  bool Changed = false;

  // Decompose function arguments first
  Function *NF = F;
  NF = visitFunctionArgs(F);
  if (NF != F) {
    //update CallGraph
    //CG->addToCallGraph(NF);
    //auto *OldNode = (*CG)[F];
    //OldNode->removeAllCalledFunctions();
    Changed = true;
  }

  if (NF->isDeclaration())
    return Changed;

  // Clear the PHINode's work on each function iteration
  PHINodeSet.clear();

  for (auto *BB : ReversePostOrderTraversal<Function *>(NF)) {
    SmallVector<Instruction*, 16> Insts;
    for (auto &I: *BB)
      Insts.push_back(&I);
    for (auto *I: Insts) {
      Changed |= visit(I);
    }
  }

  // Enhance PNINode
  Changed |= visitPHINodesAgain();

  //Changed |= visitFunction();

  return Changed;
}

void DisAggregate::addReplacementValue(Value* OrigVal, Value* NewVal) {
    mReplacementMap[OrigVal].push_back(NewVal);
    //mReferenceMap[new_val].insert(orig_val);
}

bool DisAggregate::hasReplacement(Value* Val) {
  return mReplacementMap.find(Val) != mReplacementMap.end();
}

void DisAggregate::getReplacement(Value* Val, SmallVectorImpl<Value*>& Repl,
                                  bool recursive) {
  assert(mReplacementMap.count(Val) > 0);

  SmallVectorImpl<Value*>& Vec = mReplacementMap[Val];
  for (unsigned i = 0; i < Vec.size(); ++i) {
    if (!recursive || !mReplacementMap.count(Vec[i]))
      Repl.push_back(Vec[i]);
    else
      getReplacement(Vec[i], Repl, recursive);
  }
}

Function *DisAggregate::visitFunctionArgs(Function *F) {
  if (F->isVarArg())
    return F;

  // Find the arguments to decompose.
  SetVector<Argument*> ArgsToBeDecomposed;
  for (auto &Arg:F->args()) {
    if (needToBeDecomposed(&Arg))
      ArgsToBeDecomposed.insert(&Arg);
  }

  // Nothing to work on.
  if (ArgsToBeDecomposed.empty())
      return F; 

  // Create the new function prototype.
  Function* NF = nullptr;
  SmallVector<Type*, 8> NewParams;

  // Attribute - Keep track of the parameter attributes for the arguments
  // that don't need to be decomposed. For the ones that we do decompose, the
  // parameter attributes are lost
  SmallVector<AttributeSet, 8> ArgAttrVec;
  AttributeList PAL = F->getAttributes();

  unsigned ArgNo = 0;
  for (auto &Arg:F->args()) {
    if (!ArgsToBeDecomposed.count(&Arg)) {
      NewParams.push_back(Arg.getType());
      // Keep the old attribute.
      ArgAttrVec.push_back(PAL.getParamAttributes(ArgNo));
      ArgNo++;
      continue;
    }

    SmallVector<Type*, 5> NewTys;
    getDecomposedStructTypes(Arg.getType(), NewTys);
    NewParams.insert(NewParams.end(), NewTys.begin(), NewTys.end());
    ArgAttrVec.insert(ArgAttrVec.end(), NewTys.size(), AttributeSet());
    ArgNo++;
  } 

  // Create the new function prototype.
  FunctionType* NewFTy = FunctionType::get(F->getReturnType(), 
                                            NewParams, 
                                            F->isVarArg());
  NF = Function::Create(NewFTy, F->getLinkage(), "");
  NF->takeName(F);
  NF->copyAttributesFrom(F);

  // Patch the pointer to LLVM function in debug info descriptor.
  NF->setSubprogram(F->getSubprogram());
  F->setSubprogram(nullptr);

  // Recompute the parameter attributes list based on the new arguments for
  // the function.
  NF->setAttributes(AttributeList::get(F->getContext(), PAL.getFnAttributes(),
                                       PAL.getRetAttributes(), ArgAttrVec));

  F->getParent()->getFunctionList().insert(F->getIterator(), NF); 
  F->setName(NF->getName());
  
  // Splice the body of the old function into the new function.
  NF->getBasicBlockList().splice(NF->begin(),
                                    F->getBasicBlockList());

  auto NewAI = NF->arg_begin();
  for (auto &Arg: F->args()) {
    if (ArgsToBeDecomposed.count(&Arg) <= 0) {
      NewAI->takeName(&Arg);
      Arg.replaceAllUsesWith(&*NewAI);
      // Link the old argument with the new version.
      addReplacementValue(&Arg, &*NewAI);
      ++NewAI;
      continue;
    }

    
    StructType* sty = getFirstStructType(Arg.getType());
    assert(sty && "No need to be decomposed?");
    unsigned num_elements = sty->getNumElements();

    for (unsigned k = 0; k < num_elements; ++k) {
      setStructFieldVarName(&*NewAI, &Arg, k);
      addReplacementValue(&Arg, &*NewAI);
      ++NewAI;
    }
  }

  addReplacementValue(F, NF);

  return NF;
}

bool DisAggregate::visitAllocaInst(AllocaInst &AI) {
  DEBUG(dbgs() << "visit alloca inst\n");
  AI.dump();

  Type* AllocaTy= AI.getType();
  if (!needToBeDecomposed(AllocaTy))
    return false;

  if (hasReplacement(&AI))
    return false;
  // Create a new 'alloca' instruction.
  SmallVector<Type*, 5> NewTys;
  getDecomposedStructTypes(AllocaTy, NewTys);
  assert(NewTys.size() && "No member in struct?");
  
  //clearReplacement(AI);
  for (unsigned k = 0; k < NewTys.size(); ++k) {
    Type* NewTy = cast<PointerType>(NewTys[k])->getElementType();
    AllocaInst* NewAlloca = new AllocaInst(NewTy, 0, "", &AI);
    setStructFieldVarName(NewAlloca, &AI, k);
    
    addReplacementValue(&AI, NewAlloca);
  }

  return true;
}

bool DisAggregate::visitLoadInst(LoadInst &LI) {
  DEBUG(dbgs() << "visit load inst\n");

  if (!needToBeDecomposed(&LI))
    return false;

  if (hasReplacement(&LI))
    return false;

  Value* PtrOp = LI.getPointerOperand();

  SmallVector<Value*, 5> NewOps;
  getReplacement(PtrOp, NewOps);
  assert(NewOps.size() > 0);
  
  //MDNode* MD = LI.getMetadata("dbg");
  for (unsigned i = 0; i < NewOps.size(); ++i) {
    LoadInst* NewLoad = new LoadInst(NewOps[i], "",
                                      LI.isVolatile(),
                                      &LI);
    //NewLoad->setMetadata("dbg", MD);
    setStructFieldVarName(NewLoad, &LI, i);
    addReplacementValue(&LI, NewLoad);
  }

  return true;
}

bool DisAggregate::visitStoreInst(StoreInst &SI) {
  DEBUG(dbgs() << "visit store inst\n");

  Value* Val = SI.getOperand(0);
  Value* Ptr = SI.getPointerOperand();   
  if (!needToBeDecomposed(Val))
    return false;

  SmallVector<Value*, 5> NewValues;
  getReplacement(Val, NewValues);
  SmallVector<Value*, 5> NewPtrs;
  getReplacement(Ptr, NewPtrs);
  assert(NewValues.size());
  assert(NewValues.size() == NewPtrs.size() && "Size must match!");

  // Type must match as well.
  for (unsigned i = 0; i < NewValues.size(); ++i) {
    PointerType* PTy = cast<PointerType>(NewPtrs[i]->getType());
    Type* ValTy = NewValues[i]->getType();
    assert(ValTy == PTy->getElementType());
  }

  //MDNode* MD = SI.getMetadata("dbg");
  for (unsigned i = 0; i < NewValues.size(); ++i) {
    StoreInst* NewStore = new StoreInst(NewValues[i],
                                         NewPtrs[i],
                                         SI.isVolatile(),
                                         SI.getAlignment(),
                                         &SI);
    //NewStore->setMetadata("dbg", MD);
    //addReplacementValue(SI, NewStore);
  }

  eraseInstructions(&SI);
  return true;
}

bool DisAggregate::visitGetElementPtrInst(GetElementPtrInst &GEP) {
  DEBUG(dbgs() << "visit gep inst\n");

  Value* PtrOp = GEP.getPointerOperand();
  if (!needToBeDecomposed(PtrOp))
    return false;

  if (hasReplacement(&GEP))
    return false;

  // Get the new operands.
  SmallVector<Value*, 5> NewOps;
  getReplacement(PtrOp, NewOps);
  assert(NewOps.size() > 0);
  
  Type* Ty = PtrOp->getType();
  StructType* STy = getFirstStructType(Ty);
  assert(STy && "No struct type?");
  unsigned NumElements = STy->getNumElements();
  assert(NewOps.size() == NumElements && "mismatch");

  // New indices vector.
  SmallVector<Value*, 5> NewIndices;
  Value* NewPtr = nullptr;
  
  // Process each index.
  uint64_t index = 0;
  for (auto GTI = gep_type_begin(GEP), E = gep_type_end(GEP); GTI != E; ++GTI) {
    // Indices that should be preserved.
    if (NewPtr) {
      NewIndices.push_back(GTI.getOperand());
    // Strip off the sequential types.
    } else if (!isa<StructType>(Ty)) {
      Ty = GTI.getIndexedType();
      NewIndices.push_back(GTI.getOperand());
    } else {
      // struct type
      ConstantInt* CIdx = cast<ConstantInt>(GTI.getOperand());
      index = CIdx->getZExtValue();
      assert(index < NumElements);
      NewPtr = NewOps[index];
    }
  }

  if (NewPtr) {
    // Create the new GEP instruction.
    GetElementPtrInst* NewGEP
      = GetElementPtrInst::Create(nullptr, NewPtr, 
                                  makeArrayRef(NewIndices),
                                  "",
                                  &GEP);
    //NewGEP->takeName(GEP);
    //NewGEP->setMetadata("dbg", MD);
    GEP.replaceAllUsesWith(NewGEP);
    NewGEP->takeName(&GEP);
    eraseInstructions(&GEP);
  } else  {
    // Need to decompose this GEP as well. 
    for (unsigned i = 0; i < NumElements; ++i) {
      NewPtr = NewOps[i];
      // Create the new GEP instruction.
      GetElementPtrInst* NewGep 
        = GetElementPtrInst::Create(nullptr, NewPtr, 
                                    makeArrayRef(NewIndices), 
                                    "",
                                    &GEP);
      //NewGep->setMetadata("dbg", MD); 
      setStructFieldVarName(NewGep, &GEP, i);
      addReplacementValue(&GEP, NewGep);
    }
  }

  return true;
}

bool DisAggregate::visitCallInst(CallInst &CI) {
  DEBUG(dbgs() << "visit call inst\n");

  auto *Callee = CI.getCalledFunction();
  if (!Callee)
    return false;

  SmallVector<Value*, 1> Replace;
  getReplacement(Callee, Replace);
  if (Replace.size() == 0)
    return false;

  assert(Replace.size() == 1 && "More than one replacements?");
  auto *NewCallee = cast<Function>(Replace[0]);
  
  CallSite CS(&CI);
  const AttributeList &CallPAL = CS.getAttributes();
  SmallVector<AttributeSet, 8> ArgAttrVec;
  SmallVector<Value*, 10> NewArgs;
  auto PI = NewCallee->arg_begin();
  unsigned ArgNo = 0;
  for (auto &AI:CI.arg_operands()) {
    if (!needToBeDecomposed(AI->getType())) {
      assert(PI->getType() == AI->getType() && "Diff type?");
      NewArgs.push_back(AI.get());
      ArgAttrVec.push_back(CallPAL.getParamAttributes(ArgNo));
      PI++; ArgNo++;
      continue;
    }

    // collect decomposed args
    auto* STy = getFirstStructType(AI->getType());
    auto NumElements = STy->getNumElements();
    SmallVector<Value*, 5> NewOps;
    getReplacement(AI.get(), NewOps);
    // FIXME: maybe constant arg need to be handled specially?
    // Like, UndefValue
    assert(NewOps.size() == NumElements && "Should have the same num.");
    for (unsigned i=0; i<NumElements; i++) {
      assert(NewOps[i]->getType() == PI->getType() && "Diff type?");
      NewArgs.push_back(NewOps[i]);
      ArgAttrVec.push_back(AttributeSet());
      PI++;
    }
    ArgNo++;
  }

  // Create new call
  auto *NewCall = CallInst::Create(NewCallee,
                              makeArrayRef(NewArgs),
                              "", &CI);
  CallSite NewCS(NewCall);
  NewCS.setCallingConv(CS.getCallingConv());
  NewCS.setAttributes(
      AttributeList::get(CI.getContext(), CallPAL.getFnAttributes(),
                         CallPAL.getRetAttributes(), ArgAttrVec));
  NewCS->setDebugLoc(CI.getDebugLoc());

  //update CallGraph
  //CallSite OldCS(&CI);
  //CallSite NewCS(NewCall);
  //auto *CallerNode = (*CG)[CI.getFunction()];
  //auto *NewCalleeNode = (*CG)[NewCallee];
  //CallerNode->replaceCallEdge(OldCS, NewCS, NewCalleeNode);

  if (!CI.use_empty()) {
    CI.replaceAllUsesWith(NewCall);
    NewCall->takeName(&CI);
  }
  eraseInstructions(&CI);
  return true;
}

bool DisAggregate::visitExtractValueInst(ExtractValueInst &EVI) {
  DEBUG(dbgs() << "visit extract value inst\n");

  auto *AggVal = EVI.getAggregateOperand();
  auto *Ty = AggVal->getType();
  if (!needToBeDecomposed(Ty))
    return false;
  if (hasReplacement(&EVI))
    return false;

  // Get the new operands.
  SmallVector<Value*, 5> NewOps;
  getReplacement(AggVal, NewOps);
  assert(NewOps.size() > 0);

  StructType* STy = getFirstStructType(Ty);
  assert(STy && "No struct type?");
  unsigned NumElements = STy->getNumElements();
  assert(NumElements == NewOps.size() && "mismatch?");

  Value *NewOp = nullptr;
  SmallVector<unsigned, 3> NewIndices;
  for (auto It = EVI.idx_begin(); It != EVI.idx_end(); It++) {
    if (NewOp) {
      NewIndices.push_back(*It);
    } else if (Ty->isArrayTy()) {
      Ty = cast<ArrayType>(Ty)->getElementType();
      NewIndices.push_back(*It);
    } else if (Ty->isStructTy()) {
      assert(*It < NumElements);
      NewOp = NewOps[*It];
    } else {
      assert(0 && "Other types?");
    }
  }

  if (NewOp) {
    if (NewIndices.size() == 0) {
      assert(NewOp->getType() == EVI.getType());
    } else {
      NewOp = ExtractValueInst::Create(NewOp, NewIndices, "", &EVI);
      assert(NewOp->getType() == EVI.getType());
      NewOp->takeName(&EVI);
    }
    EVI.replaceAllUsesWith(NewOp);
    eraseInstructions(&EVI);
  } else {
    // Need to decompose this EVI as well. 
    if (NewIndices.size() == 0) {
      for (unsigned i = 0; i < NumElements; ++i) {
        addReplacementValue(&EVI, NewOps[i]);
      }
    } else {
      for (unsigned i = 0; i < NumElements; ++i) {
        // Create the new EVI
        auto *NewEVI = ExtractValueInst::Create(NewOps[i], NewIndices,
                                                "", &EVI);
        //NewGep->setMetadata("dbg", MD); 
        setStructFieldVarName(NewEVI, &EVI, i);
        addReplacementValue(&EVI, NewEVI);
      }
    }
  }

  return true;
}

bool DisAggregate::visitInsertValueInst(InsertValueInst &IVI) {
  auto *AggVal = IVI.getAggregateOperand();
  auto *InsertedValue = IVI.getInsertedValueOperand();
  auto *Ty = AggVal->getType();

  if (!needToBeDecomposed(Ty)) {
    auto *Arg = dyn_cast<Argument>(InsertedValue);
    // The InsertedValue may come from a sub-struct
    if (Arg && Arg->getParent() != IVI.getFunction()) {
      SmallVector<Value*, 1> NewOps;
      getReplacement(Arg, NewOps);
      assert(NewOps.size() == 1);
      IVI.setOperand(1, NewOps[0]);
      return true;
    }
    return false;
  }

  if (hasReplacement(&IVI))
    return false;

  StructType* STy = getFirstStructType(Ty);
  assert(STy && "No struct type?");
  unsigned NumElements = STy->getNumElements();

  // Get the new operands.
  SmallVector<Value*, 5> NewOps;
  getReplacement(AggVal, NewOps);
  assert(NewOps.size() > 0);
  assert(NumElements == NewOps.size() && "mismatch?");

  Value *NewOp = nullptr;
  unsigned StructIdx = -1;
  SmallVector<unsigned, 3> NewIndices;
  for (auto It = IVI.idx_begin(); It != IVI.idx_end(); It++) {
    if (NewOp) {
      NewIndices.push_back(*It);
    } else if (Ty->isArrayTy()) {
      Ty = cast<ArrayType>(Ty)->getElementType();
      NewIndices.push_back(*It);
    } else if (Ty->isStructTy()) {
      assert(*It < NumElements);
      NewOp = NewOps[*It];
      StructIdx = *It;
    } else {
      assert(0 && "Other types?");
    }
  }

  if (NewOp) {
    assert(StructIdx != unsigned(-1) && "No cross-struct access?");
    for (unsigned i = 0; i < NumElements; ++i)
      if (i == StructIdx) {
        // Here if the InsertedValue is also a struct (sub-struct), then we
        // cannot peel one by one.
        if (NewIndices.size() == 0) {
          addReplacementValue(&IVI, InsertedValue);
        } else {
          auto *NewIVI = InsertValueInst::Create(NewOp, InsertedValue,
                                                 NewIndices, "", &IVI);
          setStructFieldVarName(NewIVI, &IVI, i);
          addReplacementValue(&IVI, NewIVI);
        }
      } else {
        addReplacementValue(&IVI, NewOps[i]);
      }
  } else {
    // Haven't got the struct
    // Get the new operands for InsertedValue.
    SmallVector<Value*, 5> NewInsertedOps;
    getReplacement(InsertedValue, NewInsertedOps);
    assert(NewInsertedOps.size() > 0);

    for (unsigned i = 0; i < NumElements; ++i) {
      auto *NewIVI = InsertValueInst::Create(NewOps[i], NewInsertedOps[i],
                                             NewIndices, "", &IVI);
      setStructFieldVarName(NewIVI, &IVI, i);
      addReplacementValue(&IVI, NewIVI);
    }
  }

  return true;
}

bool DisAggregate::visitSelectInst(SelectInst &SI) {
  auto *Ty = SI.getType();

  if (!needToBeDecomposed(Ty))
    return false;
  if (hasReplacement(&SI))
    return false;

  StructType* STy = getFirstStructType(Ty);
  assert(STy && "No struct type?");
  unsigned NumElements = STy->getNumElements();

  auto *TrueVal = SI.getTrueValue();
  auto *FalseVal = SI.getFalseValue();

  SmallVector<Value*, 5> NewTrueOps;
  getReplacement(TrueVal, NewTrueOps);

  SmallVector<Value*, 5> NewFalseOps;
  getReplacement(FalseVal, NewFalseOps);

  assert(NumElements == NewTrueOps.size() &&
         NumElements == NewFalseOps.size() );

  for (unsigned i = 0; i < NumElements; i++) {
    auto* NewSI = SelectInst::Create(SI.getCondition(), NewTrueOps[i],
                                     NewFalseOps[i], "", &SI);
    setStructFieldVarName(NewSI, &SI, i);
    addReplacementValue(&SI, NewSI);
  }

  return true;
}

bool DisAggregate::visitPHINode(PHINode &PN) {
  auto *Ty = PN.getType();

  if (!needToBeDecomposed(Ty))
    return false;
  if (hasReplacement(&PN))
    return false;

  SmallVector<Type*, 5> NewTys;
  getDecomposedStructTypes(Ty, NewTys);
  assert(NewTys.size() && "No member in struct?");

  auto NumValues = PN.getNumIncomingValues();
  for (unsigned i = 0; i < NewTys.size(); i++) {
    PHINode* NewPN = PHINode::Create(NewTys[i], NumValues, "", &PN);
    setStructFieldVarName(NewPN, &PN, i);
    addReplacementValue(&PN, NewPN);
  }
  // Need further processing
  PHINodeSet.insert(&PN);

  return true;
}

// This is the PHINode's further processing
bool DisAggregate::visitPHINodesAgain() {
  bool Changed = false;

  for (auto *PN: PHINodeSet)
    Changed |= visitPHINodeAgain(PN);

  return Changed;
}

bool DisAggregate::visitPHINodeAgain(PHINode *PN) {
  StructType* STy = getFirstStructType(PN->getType());
  assert(STy && "No struct type?");
  unsigned NumElements = STy->getNumElements();

  SmallVector<Value*, 5> NewPNs;
  getReplacement(PN, NewPNs);
  assert(NewPNs.size() == NumElements && "Don't match the num?");

  auto NumValues = PN->getNumIncomingValues();
  for (unsigned i = 0; i < NumValues; i++) {
    auto *OrigIncomingVal = PN->getIncomingValue(i);
    SmallVector<Value*, 5> NewIncomingVals;
    getReplacement(OrigIncomingVal, NewIncomingVals);
    assert(NewIncomingVals.size() == NumElements && "Don't match the num?");
    auto * IncomingBB = PN->getIncomingBlock(i);
    for (unsigned j = 0; j < NumElements; j++) {
      cast<PHINode>(NewPNs[j])->addIncoming(NewIncomingVals[j], IncomingBB);
    }
  }

  assert(eraseInstructions(PN) && "Cannot be deleted?");

  return true;
}

bool DisAggregate::visitInstruction(Instruction& I) {
  dbgs() << "visit default inst\n";
  return false;
}
