#include "EmitIR.hpp"
#include <llvm/Transforms/Utils/ModuleUtils.h>

#define self (*this)

using namespace asg;

EmitIR::EmitIR(Obj::Mgr &mgr, llvm::LLVMContext &ctx, llvm::StringRef mid)
    : mMgr(mgr), mMod(mid, ctx), mCtx(ctx), mIntTy(llvm::Type::getInt32Ty(ctx)), mCurIrb(std::make_unique<llvm::IRBuilder<>>(ctx)), mCtorTy(llvm::FunctionType::get(llvm::Type::getVoidTy(ctx), false))
{
}

llvm::Module &
EmitIR::operator()(asg::TranslationUnit *tu)
{
    for (auto &&i : tu->decls)
        self(i);
    return mMod;
}

//==============================================================================
// 类型
//==============================================================================

llvm::Type *
EmitIR::operator()(const Type *type)
{
    if (type->texp == nullptr)
    {
        switch (type->spec)
        {
        case Type::Spec::kInt:
            return llvm::Type::getInt32Ty(mCtx);
        // TODO: 在此添加对更多基础类型的处理
        case Type::Spec::kVoid:
            return llvm::Type::getVoidTy(mCtx);
        default:
            ABORT();
        }
    }

    Type subt;
    subt.spec = type->spec;
    subt.qual = type->qual;
    subt.texp = type->texp->sub;

    // TODO: 在此添加对指针类型、数组类型和函数类型的处理

    if (auto p = type->texp->dcst<PointerType>())
    {
        auto subty = llvm::Type::getInt32Ty(mCtx);
        return llvm::PointerType::get(subty, 0);
    }

    if (auto current = type->texp->dcst<ArrayType>())
    {
        std::vector<unsigned> dimensions;
        do
        {
            dimensions.push_back(current->len);
            current = current->sub->dcst<ArrayType>();
        } while (current);

        llvm::Type *elementType = llvm::Type::getInt32Ty(mCtx);            // 默认元素类型为 int32
        for (auto it = dimensions.rbegin(); it != dimensions.rend(); ++it) // 从后往前遍历
            elementType = llvm::ArrayType::get(elementType, *it);          // 逐层构造数组类型

        return llvm::cast<llvm::ArrayType>(elementType);
    }

    if (auto p = type->texp->dcst<FunctionType>())
    {
        std::vector<llvm::Type *> pty;
        // TODO: 在此添加对函数参数类型的处理
        for (auto &&param : p->params)
            pty.push_back(self(param));
        return llvm::FunctionType::get(self(&subt), std::move(pty), false);
    }

    ABORT();
}

//==============================================================================
// 表达式
//==============================================================================

llvm::Value *
EmitIR::operator()(Expr *obj)
{
    // TODO: 在此添加对更多表达式处理的跳转
    if (auto p = obj->dcst<IntegerLiteral>())
        return self(p);

    if (auto p = obj->dcst<BinaryExpr>())
        return self(p);

    if (auto p = obj->dcst<ImplicitCastExpr>())
        return self(p);

    if (auto p = obj->dcst<DeclRefExpr>())
        return self(p);

    if (auto p = obj->dcst<ParenExpr>())
        return self(p);

    if (auto p = obj->dcst<UnaryExpr>())
        return self(p);

    if (auto p = obj->dcst<CallExpr>())
        return self(p);

    if (auto p = obj->dcst<InitListExpr>())
        return self(p);

    ABORT();
}

llvm::Constant *
EmitIR::operator()(IntegerLiteral *obj)
{
    return llvm::ConstantInt::get(self(obj->type), obj->val);
}

// TODO: 在此添加对更多表达式类型的处理

llvm::Value *
EmitIR::operator()(ImplicitCastExpr *obj)
{
    auto sub = self(obj->sub);

    auto &irb = *mCurIrb;
    switch (obj->kind)
    {
    case ImplicitCastExpr::kLValueToRValue:
    { // 左值到右值
        auto ty = self(obj->sub->type);
        auto loadVal = irb.CreateLoad(ty, sub);
        return loadVal;
    }
    case ImplicitCastExpr::kIntegralCast:
    { // 整数类型转换
        auto ty = self(obj->type);
        return irb.CreateIntCast(sub, ty, true);
    }
    case ImplicitCastExpr::kArrayToPointerDecay:
    { // 数组到指针的转换
        return sub;
    }
    case ImplicitCastExpr::kFunctionToPointerDecay:
    { // 函数到指针的转换
        return sub;
    }
    case ImplicitCastExpr::kNoOp:
    { // 无操作
        return sub;
    }
    default:
        ABORT();
    }
}

llvm::Value *
EmitIR::operator()(DeclRefExpr *obj)
{
    // 在LLVM IR层面，左值体现为返回指向值的指针
    // 在ImplicitCastExpr::kLValueToRValue中发射load指令从而变成右值
    return reinterpret_cast<llvm::Value *>(obj->decl->any);
}

llvm::Value *
EmitIR::operator()(ParenExpr *obj)
{
    return self(obj->sub);
}

llvm::Value *
EmitIR::operator()(BinaryExpr *obj) // 二元表达式
{
    llvm::Value *lftVal, *rhtVal;

    lftVal = self(obj->lft);

    auto &irb = *mCurIrb;
    rhtVal = self(obj->rht);
    switch (obj->op)
    {
    case BinaryExpr::kAdd:
        return irb.CreateAdd(lftVal, rhtVal);
    case BinaryExpr::kSub:
        return irb.CreateSub(lftVal, rhtVal);
    case BinaryExpr::kMul:
        return irb.CreateMul(lftVal, rhtVal);
    case BinaryExpr::kDiv:
        return irb.CreateSDiv(lftVal, rhtVal);
    case BinaryExpr::kMod:
        return irb.CreateSRem(lftVal, rhtVal);

    case BinaryExpr::kGt:
        return irb.CreateICmpSGT(lftVal, rhtVal);
    case BinaryExpr::kLt:
        return irb.CreateICmpSLT(lftVal, rhtVal);
    case BinaryExpr::kGe:
        return irb.CreateICmpSGE(lftVal, rhtVal);
    case BinaryExpr::kLe:
        return irb.CreateICmpSLE(lftVal, rhtVal);
    case BinaryExpr::kEq:
        return irb.CreateICmpEQ(lftVal, rhtVal);
    case BinaryExpr::kNe:
        return irb.CreateICmpNE(lftVal, rhtVal);
    case BinaryExpr::kAssign:
        return irb.CreateStore(rhtVal, lftVal); // "="
    case BinaryExpr::kComma:
        return rhtVal; // ","
    case BinaryExpr::kIndex:
    {
        return irb.CreateInBoundsGEP(self(obj->type), lftVal, llvm::ArrayRef<llvm::Value *>{rhtVal});
    }
    case BinaryExpr::kAnd:
    case BinaryExpr::kOr:
    {
        llvm::BasicBlock *curBlock = irb.GetInsertBlock();
        llvm::Function *func = curBlock->getParent();

        bool isAnd = (obj->op == BinaryExpr::kAnd);
        const char *opName = isAnd ? "land" : "lor";

        llvm::BasicBlock *rhsBlock = llvm::BasicBlock::Create(mCtx, std::string(opName) + ".rhs", func);
        llvm::BasicBlock *endBlock = llvm::BasicBlock::Create(mCtx, std::string(opName) + ".end", func);

        auto convertToBool = [&](llvm::Value *val)
        {
            return val->getType()->isIntegerTy(32) ? irb.CreateICmpNE(val, irb.getInt32(0), "isNonZero") : val;
        };

        llvm::Value *left_result = convertToBool(self(obj->lft));
        irb.CreateCondBr(left_result, isAnd ? rhsBlock : endBlock, isAnd ? endBlock : rhsBlock);
        llvm::BasicBlock *leftEnd = irb.GetInsertBlock();

        irb.SetInsertPoint(rhsBlock);
        llvm::Value *right_result = convertToBool(self(obj->rht));
        irb.CreateBr(endBlock);
        llvm::BasicBlock *rightEnd = irb.GetInsertBlock();

        irb.SetInsertPoint(endBlock);
        llvm::PHINode *phi = irb.CreatePHI(llvm::Type::getInt1Ty(mCtx), 2, "merge");
        phi->addIncoming(irb.getInt1(!isAnd), leftEnd);
        phi->addIncoming(right_result, rightEnd);
        return phi;
    }
    default:
        ABORT();
    }
}

llvm::Value *
EmitIR::operator()(UnaryExpr *obj) // 一元表达式：非!和取负-
{
    auto val = self(obj->sub);
    auto &irb = *mCurIrb;
    switch (obj->op)
    {
    case UnaryExpr::kNot:
    {
        llvm::Value *result = val;
        if (result->getType()->isIntegerTy(32))
        { // 这里是为了处理非0为真的情况
            result = irb.CreateICmpNE(result, irb.getInt32(0), "isNonZero");
        }
        return irb.CreateNot(result);
    }

    case UnaryExpr::kNeg:
        return irb.CreateNeg(val);

    case UnaryExpr::kPos: // 正号，注意这里一定要处理
        return val;
    default:
        ABORT();
    }
}

llvm::Value *
EmitIR::operator()(CallExpr *obj) // 函数调用
{
    auto func = reinterpret_cast<llvm::Function *>(self(obj->head));
    std::vector<llvm::Value *> args;
    for (auto &&arg : obj->args)
        args.push_back(self(arg));

    return mCurIrb->CreateCall(func, args);
}

llvm::Value *
EmitIR::operator()(InitListExpr *obj)
{
    llvm::ArrayType *ty = reinterpret_cast<llvm::ArrayType *>(self(obj->type));
    auto &irb = *mCurIrb;
    llvm::Value *pointer = irb.CreateAlloca(ty, nullptr);
    for (size_t i = 0; i < obj->list.size(); ++i)
    {
        if (obj->list[i]->dcst<ImplicitInitExpr>())
        {
            return llvm::Constant::getNullValue(ty);
        }
        if (self(obj->list[i]->type)->isIntegerTy())
        { // 第一层
            auto val = self(obj->list[i]);
            std::vector<llvm::Value *> idxList{irb.getInt64(0), irb.getInt64((int)i)};
            llvm::Value *element = irb.CreateInBoundsGEP(ty, pointer, idxList);
            irb.CreateStore(val, element);
        }
        else
        {                                                                                            // 第二层
            auto pointer1D = self(obj->list[i]);                                                     // 获得第一层指针
            llvm::ArrayType *innerArrayType = llvm::dyn_cast<llvm::ArrayType>(ty->getElementType()); // 获得第一层类型
            for (size_t j = 0; j < innerArrayType->getNumElements(); j++)
            { // 遍历第一层元素
                std::vector<llvm::Value *> idxList1D{irb.getInt64(0), irb.getInt64((int)j)};
                llvm::Value *element1D = irb.CreateInBoundsGEP(innerArrayType, pointer1D, idxList1D);
                auto val = irb.CreateLoad(llvm::Type::getInt32Ty(mCtx), element1D); // 取出值
                std::vector<llvm::Value *> idxList{irb.getInt64(0), irb.getInt64((int)i), irb.getInt64((int)j)};
                llvm::Value *element = irb.CreateInBoundsGEP(ty, pointer, idxList);
                irb.CreateStore(val, element); // 存入值
            }
        }
    }
    return pointer;
}

//==============================================================================
// 语句
//==============================================================================

void EmitIR::operator()(Stmt *obj)
{
    // TODO: 在此添加对更多Stmt类型的处理的跳转

    if (auto p = obj->dcst<CompoundStmt>())
        return self(p);

    if (auto p = obj->dcst<ReturnStmt>())
        return self(p);

    if (auto p = obj->dcst<DeclStmt>())
        return self(p);

    if (auto p = obj->dcst<ExprStmt>())
        return self(p);

    if (auto p = obj->dcst<NullStmt>())
        return self(p);

    if (auto p = obj->dcst<IfStmt>())
        return self(p);

    if (auto p = obj->dcst<WhileStmt>())
        return self(p);

    if (auto p = obj->dcst<BreakStmt>())
        return self(p);

    if (auto p = obj->dcst<ContinueStmt>())
        return self(p);

    ABORT();
}

// TODO: 在此添加对更多Stmt类型的处理

void EmitIR::operator()(CompoundStmt *obj)
{
    // TODO: 可以在此添加对符号重名的处理
    for (auto &&stmt : obj->subs)
        self(stmt);
}

void EmitIR::operator()(ReturnStmt *obj)
{
    auto &irb = *mCurIrb;

    llvm::Value *retVal;
    if (!obj->expr)
        retVal = nullptr;
    else
        retVal = self(obj->expr);

    mCurIrb->CreateRet(retVal);

    auto exitBb = llvm::BasicBlock::Create(mCtx, "return_exit", mCurFunc);
    mCurIrb->SetInsertPoint(exitBb);
}

void EmitIR::operator()(DeclStmt *obj)
{
    for (auto &&decl : obj->decls)
        self(decl);
}

void EmitIR::operator()(ExprStmt *obj)
{
    self(obj->expr);
}

void EmitIR::operator()(NullStmt *obj)
{
    // do nothing
}

llvm::BasicBlock *breakTarget = nullptr;
llvm::BasicBlock *continueTarget = nullptr;

void EmitIR::operator()(BreakStmt *obj)
{
    auto &irb = *mCurIrb;
    if (breakTarget)
        irb.CreateBr(breakTarget);
}

void EmitIR::operator()(ContinueStmt *obj)
{
    auto &irb = *mCurIrb;
    if (continueTarget)
        irb.CreateBr(continueTarget);
}

void EmitIR::operator()(WhileStmt *obj)
{
    auto &irb = *mCurIrb;
    llvm::Function *func = irb.GetInsertBlock()->getParent();
    llvm::BasicBlock *condition = llvm::BasicBlock::Create(mCtx, "while.cond", func);
    llvm::BasicBlock *body = llvm::BasicBlock::Create(mCtx, "while.body", func);
    llvm::BasicBlock *end = llvm::BasicBlock::Create(mCtx, "while.end", func);

    llvm::BasicBlock *prevBreak = breakTarget;
    llvm::BasicBlock *prevContinue = continueTarget;
    breakTarget = end;
    continueTarget = condition;

    irb.CreateBr(condition);
    irb.SetInsertPoint(condition);
    auto condVal = self(obj->cond);
    if (condVal->getType()->isIntegerTy(32))
    {
        condVal = irb.CreateICmpNE(condVal, irb.getInt32(0), "isNonZero");
    }
    irb.CreateCondBr(condVal, body, end);

    irb.SetInsertPoint(body);
    self(obj->body);
    if (!irb.GetInsertBlock()->getTerminator())
    {
        irb.CreateBr(condition);
    }

    irb.SetInsertPoint(end);
    breakTarget = prevBreak;
    continueTarget = prevContinue;
}

void EmitIR::operator()(IfStmt *obj)
{
    auto &irb = *mCurIrb;
    llvm::Function *func = irb.GetInsertBlock()->getParent();
    llvm::BasicBlock *thenBB = llvm::BasicBlock::Create(mCtx, "if.then", func);
    llvm::BasicBlock *endBB = llvm::BasicBlock::Create(mCtx, "if.end", func);

    auto condVal = self(obj->cond);
    if (condVal->getType()->isIntegerTy(32))
    {
        condVal = irb.CreateICmpNE(condVal, irb.getInt32(0), "isNonZero");
    }

    if (obj->else_)
    {
        llvm::BasicBlock *elseBB = llvm::BasicBlock::Create(mCtx, "if.else", func);
        irb.CreateCondBr(condVal, thenBB, elseBB);
        irb.SetInsertPoint(thenBB);
        self(obj->then);
        if (!irb.GetInsertBlock()->getTerminator())
            irb.CreateBr(endBB);
        irb.SetInsertPoint(elseBB);
        self(obj->else_);
        if (!irb.GetInsertBlock()->getTerminator())
            irb.CreateBr(endBB);
    }
    else
    {
        irb.CreateCondBr(condVal, thenBB, endBB);
        irb.SetInsertPoint(thenBB);
        self(obj->then);
        if (!irb.GetInsertBlock()->getTerminator())
            irb.CreateBr(endBB);
    }

    irb.SetInsertPoint(endBB);
}

//==============================================================================
// 声明
//==============================================================================

void EmitIR::operator()(Decl *obj)
{
    // TODO: 添加变量声明处理的跳转

    if (auto p = obj->dcst<VarDecl>())
        return self(p);

    if (auto p = obj->dcst<FunctionDecl>())
        return self(p);

    ABORT();
}

// TODO: 添加变量声明的处理

void EmitIR::trans_init(llvm::Value *val, Expr *obj) // 这个函数是用来处理变量初始化的，val是变量的地址，obj是初始化表达式
{
    auto &irb = *mCurIrb;

    // 仅处理整数字面量的初始化
    if (auto p = obj->dcst<IntegerLiteral>())
    {
        auto initVal = llvm::ConstantInt::get(self(p->type), p->val);
        irb.CreateStore(initVal, val);
        return;
    }

    if (auto p = obj->dcst<UnaryExpr>())
    {
        auto initVal = self(p);
        irb.CreateStore(initVal, val);
        return;
    }

    if (auto p = obj->dcst<BinaryExpr>())
    {
        auto initVal = self(p);
        irb.CreateStore(initVal, val);
        return;
    }

    if (auto p = obj->dcst<DeclRefExpr>())
    {
        auto initVal = self(p);
        irb.CreateStore(initVal, val);
        return;
    }

    if (auto p = obj->dcst<CallExpr>())
    {
        auto initVal = self(p);
        irb.CreateStore(initVal, val);
        return;
    }

    if (auto p = obj->dcst<ParenExpr>())
    {
        auto initVal = self(p);
        irb.CreateStore(initVal, val);
        return;
    }

    if (auto p = obj->dcst<InitListExpr>())
    {
        if (p->list.empty() || p->list[0]->dcst<ImplicitInitExpr>())
            return;

        llvm::Value *pointer = self(p);
        auto *arrType = llvm::dyn_cast<llvm::ArrayType>(val->getType()->getArrayElementType());
        if (!arrType)
            ABORT(); // 确保 val 是数组指针

        std::vector<llvm::Value *> idxBase{irb.getInt64(0)};
        auto elementTy = arrType->getElementType();

        // 递归处理多维数组
        std::function<void(llvm::ArrayType *, std::vector<llvm::Value *>)> storeElements =
            [&](llvm::ArrayType *ty, std::vector<llvm::Value *> idxList)
        {
            int numElements = ty->getNumElements();
            auto *subTy = ty->getElementType();

            for (int i = 0; i < numElements; i++)
            {
                auto idx = idxList;
                idx.push_back(irb.getInt64(i));

                if (auto *innerArrayTy = llvm::dyn_cast<llvm::ArrayType>(subTy))
                {
                    storeElements(innerArrayTy, idx);
                }
                else
                {
                    llvm::Value *srcElement = irb.CreateInBoundsGEP(ty, pointer, idx);
                    llvm::Value *dstElement = irb.CreateInBoundsGEP(ty, val, idx);
                    llvm::Value *initVal = irb.CreateLoad(irb.getInt32Ty(), srcElement);
                    irb.CreateStore(initVal, dstElement);
                }
            }
        };

        storeElements(arrType, idxBase);
        return;
    }

    if (auto p = obj->dcst<ImplicitInitExpr>())
    {
        auto initVal = self(p);
        irb.CreateStore(initVal, val);
        return;
    }
    // 如果表达式不是整数字面量，则中断编译
    ABORT();
}

void EmitIR::operator()(VarDecl *obj) // 考虑全局变量和局部变量，不能用islocal判断
{
    auto &irb = *mCurIrb;
    llvm::BasicBlock *entryBb = irb.GetInsertBlock();
    if (entryBb)
    { // 局部变量
        auto ty = self(obj->type);
        auto alloca = irb.CreateAlloca(ty, nullptr, obj->name);
        obj->any = alloca;

        if (obj->init == nullptr)
            return;

        auto initBb = llvm::Constant::getNullValue(ty);
        irb.CreateStore(initBb, alloca);
        trans_init(alloca, obj->init);
    }
    else
    {                                           // 全局变量
        auto ty = llvm::Type::getInt32Ty(mCtx); // 直接使用 LLVM 的 int32 类型
        auto gvar = new llvm::GlobalVariable(
            mMod, ty, false, llvm::GlobalVariable::ExternalLinkage, nullptr, obj->name);

        obj->any = gvar;

        // 默认初始化为 0
        gvar->setInitializer(llvm::ConstantInt::get(ty, 0));

        if (obj->init == nullptr)
            return;

        // 创建构造函数用于初始化
        mCurFunc = llvm::Function::Create(
            mCtorTy, llvm::GlobalVariable::PrivateLinkage, "ctor_" + obj->name, mMod);
        llvm::appendToGlobalCtors(mMod, mCurFunc, 65535);

        auto entryBb = llvm::BasicBlock::Create(mCtx, "entry", mCurFunc);
        mCurIrb->SetInsertPoint(entryBb);
        trans_init(gvar, obj->init);
        mCurIrb->CreateRet(nullptr);
    }
}

void EmitIR::operator()(FunctionDecl *obj)
{
    // 创建函数
    auto fty = llvm::dyn_cast<llvm::FunctionType>(self(obj->type));
    auto func = llvm::Function::Create(
        fty, llvm::GlobalVariable::ExternalLinkage, obj->name, mMod);

    obj->any = func;

    if (obj->body == nullptr)
        return;
    auto entryBb = llvm::BasicBlock::Create(mCtx, "entry", func);
    mCurIrb->SetInsertPoint(entryBb);
    auto &entryIrb = *mCurIrb;

    // TODO: 添加对函数参数的处理
    auto argIt = func->arg_begin();
    for (auto &&param : obj->params)
    {
        auto arg = &*argIt++;
        arg->setName(param->name);
        auto alloca = entryIrb.CreateAlloca(self(param->type), nullptr, param->name);
        entryIrb.CreateStore(arg, alloca);
        param->any = alloca;
    }
    // 翻译函数体
    mCurFunc = func;
    self(obj->body);
    auto &exitIrb = *mCurIrb;

    if (fty->getReturnType()->isVoidTy())
        exitIrb.CreateRetVoid();
    else
        exitIrb.CreateUnreachable();
}
