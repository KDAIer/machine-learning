#include "Ast2Asg.hpp"
#include <unordered_map>

#define self (*this)

namespace asg {

// 符号表，保存当前作用域的所有声明
struct Ast2Asg::Symtbl : public std::unordered_map<std::string, Decl*>
{
  Ast2Asg& m;
  Symtbl* mPrev;

  Symtbl(Ast2Asg& m)
    : m(m)
    , mPrev(m.mSymtbl)
  {
    m.mSymtbl = this;
  }

  ~Symtbl() { m.mSymtbl = mPrev; }

  Decl* resolve(const std::string& name);
};

Decl*
Ast2Asg::Symtbl::resolve(const std::string& name)
{
  auto iter = find(name);
  if (iter != end())
    return iter->second;
  ASSERT(mPrev != nullptr); // 标识符未定义
  return mPrev->resolve(name);
}

TranslationUnit*
Ast2Asg::operator()(ast::TranslationUnitContext* ctx)
{
  auto ret = make<asg::TranslationUnit>();
  if (ctx == nullptr)
    return ret;

  Symtbl localDecls(self);

  for (auto&& i : ctx->externalDeclaration()) {
    if (auto p = i->declaration()) {
      auto decls = self(p);
      ret->decls.insert(ret->decls.end(),
                        std::make_move_iterator(decls.begin()),
                        std::make_move_iterator(decls.end()));
    }

    else if (auto p = i->functionDefinition()) {
      auto funcDecl = self(p);
      ret->decls.push_back(funcDecl);

      // 添加到声明表
      localDecls[funcDecl->name] = funcDecl;
    }

    else
      ABORT();
  }

  return ret;
}

//==============================================================================
// 类型
//==============================================================================

Ast2Asg::SpecQual
Ast2Asg::operator()(ast::DeclarationSpecifiersContext* ctx)
{
  SpecQual ret = { Type::Spec::kINVALID, Type::Qual() };

  for (auto&& i : ctx->declarationSpecifier()) {
    if (auto p = i->typeSpecifier()) {
      if (ret.first == Type::Spec::kINVALID) {
        if (p->Int())
          ret.first = Type::Spec::kInt;
        else if (p->Void())
          ret.first = Type::Spec::kVoid;
        else if (p->Const())
          ret.second.const_ = true;
        else
          ABORT(); // 未知的类型说明符
      }

      else
        ABORT(); // 未知的类型说明符
    }

    else
      ABORT();
  }

  return ret;
}

std::pair<TypeExpr*, std::string>
Ast2Asg::operator()(ast::DeclaratorContext* ctx, TypeExpr* sub)
{
  return self(ctx->directDeclarator(), sub);
}

static int
eval_arrlen(Expr* expr)
{
  if (auto p = expr->dcst<IntegerLiteral>())
    return p->val;

  if (auto p = expr->dcst<DeclRefExpr>()) {
    if (p->decl == nullptr)
      ABORT();

    auto var = p->decl->dcst<VarDecl>();
    if (!var || !var->type->qual.const_)
      ABORT(); // 数组长度必须是编译期常量

    switch (var->type->spec) {
      case Type::Spec::kChar:
      case Type::Spec::kInt:
      case Type::Spec::kLong:
      case Type::Spec::kLongLong:
        return eval_arrlen(var->init);

      default:
        ABORT(); // 长度表达式必须是数值类型
    }
  }

  if (auto p = expr->dcst<UnaryExpr>()) {
    auto sub = eval_arrlen(p->sub);

    switch (p->op) {
      case UnaryExpr::kPos:
        return sub;

      case UnaryExpr::kNeg:
        return -sub;

      default:
        ABORT();
    }
  }

  if (auto p = expr->dcst<BinaryExpr>()) {
    auto lft = eval_arrlen(p->lft);
    auto rht = eval_arrlen(p->rht);

    switch (p->op) {
      case BinaryExpr::kAdd:
        return lft + rht;

      case BinaryExpr::kSub:
        return lft - rht;

      default:
        ABORT();
    }
  }

  if (auto p = expr->dcst<InitListExpr>()) {
    if (p->list.empty())
      return 0;
    return eval_arrlen(p->list[0]);
  }

  ABORT();
}

std::pair<TypeExpr*, std::string>
Ast2Asg::operator()(ast::DirectDeclaratorContext* ctx, TypeExpr* sub)
{
  if (auto p = ctx->Identifier())
    return { sub, p->getText() };

  if (ctx->LeftBracket()) {
    auto arrayType = make<ArrayType>();
    arrayType->sub = sub;

    if (auto p = ctx->assignmentExpression())
      arrayType->len = eval_arrlen(self(p));
    else
      arrayType->len = ArrayType::kUnLen;

    return self(ctx->directDeclarator(), arrayType);
  }

  ABORT();
}

//==============================================================================
// 表达式
//==============================================================================

Expr*
Ast2Asg::operator()(ast::ExpressionContext* ctx)
{
  auto list = ctx->assignmentExpression();
  Expr* ret = self(list[0]);

  for (unsigned i = 1; i < list.size(); ++i) {
    auto node = make<BinaryExpr>();
    node->op = node->kComma;
    node->lft = ret;
    node->rht = self(list[i]);
    ret = node;
  }

  return ret;
}

Expr*
Ast2Asg::operator()(ast::AssignmentExpressionContext* ctx)
{
  if (auto p = ctx->additiveExpression())
    return self(p);

  auto ret = make<BinaryExpr>();
  ret->op = ret->kAssign;
  ret->lft = self(ctx->unaryExpression());
  ret->rht = self(ctx->assignmentExpression());
  return ret;
}


Expr*
Ast2Asg::operator()(ast::ParenExpressionContext* ctx)
{
  auto p = make<ParenExpr>();
  auto sub = ctx->children;
  p->sub = self(dynamic_cast<ast::AdditiveExpressionContext*>(sub[1]));
  return p;
}

Expr*
Ast2Asg::operator()(ast::MultiplicativeExpressionContext* ctx)
{
  auto children = ctx->children;
  // assert(dynamic_cast<ast::UnaryExpressionContext*>(children[0]));

  auto t1 = dynamic_cast<ast::UnaryExpressionContext*>(children[0]);
  auto t2 = dynamic_cast<ast::ParenExpressionContext*>(children[0]);

  Expr* ret = t1 != nullptr ? self(t1) : self(t2);

  for (unsigned i = 1; i < children.size(); ++i) {
    auto node = make<BinaryExpr>();

    auto token = dynamic_cast<antlr4::tree::TerminalNode*>(children[i])
                   ->getSymbol()
                   ->getType();
    switch (token) {
      case ast::Star:
        node->op = node->kMul;
        break;

      case ast::Slash:
        node->op = node->kDiv;
        break;

      case ast::Percent:
        node->op = node->kMod;
        break;

      default:
        ABORT();
    }

    node->lft = ret;
    if (auto p = dynamic_cast<ast::UnaryExpressionContext*>(children[++i]))
      node->rht = self(p);
    else
      node->rht = self(dynamic_cast<ast::ParenExpressionContext*>(children[++i]));
    ret = node;
  }

  return ret;
}


Expr*
Ast2Asg::operator()(ast::AdditiveExpressionContext* ctx)
{
  auto children = ctx->children;
  // assert(dynamic_cast<ast::UnaryExpressionContext*>(children[0]));
  Expr* ret =
    self(dynamic_cast<ast::MultiplicativeExpressionContext*>(children[0]));

  for (unsigned i = 1; i < children.size(); ++i) {
    auto node = make<BinaryExpr>();

    auto token = dynamic_cast<antlr4::tree::TerminalNode*>(children[i])
                   ->getSymbol()
                   ->getType();
    switch (token) {
      case ast::Plus:
        node->op = node->kAdd;
        break;

      case ast::Minus:
        node->op = node->kSub;
        break;

      default:
        ABORT();
    }

    node->lft = ret;
    node->rht =
      self(dynamic_cast<ast::MultiplicativeExpressionContext*>(children[++i]));
    ret = node;
  }

  return ret;
}

Expr*
Ast2Asg::operator()(ast::RelExprContext* ctx)
{
  auto children = ctx->children;
  // 处理第一个关系表达式
  Expr* ret = self(dynamic_cast<ast::UnaryExpressionContext*>(children[0]));

  for (unsigned i = 1; i < children.size(); ++i) {
    auto node = make<BinaryExpr>();

    // 获取当前节点的操作符
    auto token = dynamic_cast<antlr4::tree::TerminalNode*>(children[i])->getSymbol()->getType();

    switch (token) {
      case ast::Less:
        node->op = node->kLt;
        break;
      case ast::Lessequal:
        node->op = node->kLe;
        break;
      case ast::Greater:
        node->op = node->kGt;
        break;
      case ast::Greaterequal:
        node->op = node->kGe;
        break;
      case ast::Equalequal:
        node->op = node->kEq;
        break;
      case ast::Exclaimequal:
        node->op = node->kNe;
        break;
      default:
        ABORT();
    }

    // 连接左操作数（之前的关系表达式）和右操作数（下一个加法表达式）
    node->lft = ret;
    node->rht = self(dynamic_cast<ast::UnaryExpressionContext*>(children[++i]));
    ret = node;
  }

  return ret;
}

//条件表达式中层---------------------------------------------------
Expr*
Ast2Asg::operator()(ast::Condition_midContext* ctx)
{
  auto children = ctx->children;
  Expr* ret = self(dynamic_cast<ast::RelExprContext*>(children[0]));

  for (unsigned i = 1; i < children.size(); ++i) {
    auto node = make<BinaryExpr>();

    auto token = dynamic_cast<antlr4::tree::TerminalNode*>(children[i])
                   ->getSymbol()
                   ->getType();
    switch (token) {
      case ast::Ampamp:
        node->op = node->kAnd;
        break;

      default:
        ABORT();
    }

    node->lft = ret;
    node->rht = self(dynamic_cast<ast::RelExprContext*>(children[++i]));
    ret = node;
  }

  return ret;
}

//条件表达式高层---------------------------------------------------
Expr*
Ast2Asg::operator()(ast::Condition_highContext* ctx)
{
  auto children = ctx->children;
  Expr* ret = self(dynamic_cast<ast::Condition_midContext*>(children[0]));

  for (unsigned i = 1; i < children.size(); ++i) {
    auto node = make<BinaryExpr>();

    auto token = dynamic_cast<antlr4::tree::TerminalNode*>(children[i])
                   ->getSymbol()
                   ->getType();
    switch (token) {
      case ast::Pipepipe:
        node->op = node->kOr;
        break;

      default:
        ABORT();
    }

    node->lft = ret;
    node->rht = self(dynamic_cast<ast::Condition_midContext*>(children[++i]));
    ret = node;
  }

  return ret;
}

Expr*
Ast2Asg::operator()(ast::UnaryExpressionContext* ctx)
{
  if (auto p = ctx->postfixExpression())
    return self(p); 

  if (auto p = ctx->parenExpression()) //注意这里对括号的处理（处理024测试样例）
    return self(p);

  auto ret = make<UnaryExpr>();

  switch (
    dynamic_cast<antlr4::tree::TerminalNode*>(ctx->unaryOperator()->children[0])
      ->getSymbol()
      ->getType()) {
    case ast::Plus:
      ret->op = ret->kPos;
      break;

    case ast::Minus:
      ret->op = ret->kNeg;
      break;

    case ast::Exclaim:
      ret->op = ret->kNot;
      break;
    
    default:
      ABORT();
  }

  ret->sub = self(ctx->unaryExpression());

  return ret;
}

Expr*
Ast2Asg::operator()(ast::PostfixExpressionContext* ctx)
{
  auto children = ctx->children;
  auto sub = self(dynamic_cast<ast::PrimaryExpressionContext*>(children[0]));
  return sub;
}

Expr*
Ast2Asg::operator()(ast::PrimaryExpressionContext* ctx)
{

  if (auto p = ctx->Identifier()) {
    auto name = p->getText();
    auto ret = make<DeclRefExpr>();
    ret->decl = mSymtbl->resolve(name);
    return ret;
  }

  if (auto p = ctx->Constant()) {
    auto text = p->getText();

    auto ret = make<IntegerLiteral>();

    ASSERT(!text.empty());
    if (text[0] != '0')
      ret->val = std::stoll(text);

    else if (text.size() == 1)
      ret->val = 0;

    else if (text[1] == 'x' || text[1] == 'X')
      ret->val = std::stoll(text.substr(2), nullptr, 16);

    else
      ret->val = std::stoll(text.substr(1), nullptr, 8);

    return ret;
  }

  if (auto p = ctx->array()) { 
    auto children = p->children;
    auto name = children[0]->getText();
    // 创建声明引用表达式
    auto dec = make<DeclRefExpr>();
    dec->decl = mSymtbl->resolve(name);
    // 初始化返回表达式
    auto ret = make<BinaryExpr>();
    ret->op = BinaryExpr::kIndex;
    ret->lft = dec;
    // 遍历数组索引
    for (unsigned i = 1; i < children.size(); ++i) {
        auto currentChild = children[i];
        // 只处理赋值表达式节点
        if (auto exprCtx = dynamic_cast<ast::AssignmentExpressionContext*>(currentChild)) {
            auto indexExpr = self(exprCtx);
            auto newIndexNode = make<BinaryExpr>();
            newIndexNode->op = BinaryExpr::kIndex;
            if (i == 2) {
                newIndexNode->lft = dec;
            } else {
                newIndexNode->lft = ret;
            }
            newIndexNode->rht = indexExpr;
            ret = newIndexNode;
        }
    }
    return ret;
  }

  if(auto p = ctx->functioncall()){
    auto children = p->children;
    // 创建函数调用表达式
    auto ret = make<CallExpr>();
    // 处理函数名
    {
        auto funcName = children[0]->getText();
        auto dec = make<DeclRefExpr>();
        dec->decl = mSymtbl->resolve(funcName);
        ret->head = dec;
    }
    // 处理参数列表
    for (unsigned i = 1; i < children.size(); ++i) {
        auto currentChild = children[i];
        // 只处理加法表达式节点（参数）
        if (auto exprCtx = dynamic_cast<ast::AdditiveExpressionContext*>(currentChild)) {
            ret->args.push_back(self(exprCtx));
        }
    }
    return ret;
  } 
  ABORT();
}

Expr*
Ast2Asg::operator()(ast::InitializerContext* ctx)
{
  if (auto p = ctx->assignmentExpression())
    return self(p);

  auto ret = make<InitListExpr>();

  if (auto p = ctx->initializerList()) {
    for (auto&& i : p->initializer()) {
      // 将初始化列表展平
      auto expr = self(i);
      if (auto p = expr->dcst<InitListExpr>()) {
        for (auto&& sub : p->list)
          ret->list.push_back(sub);
      } else {
        ret->list.push_back(expr);
      }
    }
  }

  return ret;
}

//==============================================================================
// 语句
//==============================================================================

Stmt*
Ast2Asg::operator()(ast::StatementContext* ctx)
{
  if (auto p = ctx->compoundStatement())
    return self(p);

  if (auto p = ctx->expressionStatement())
    return self(p);

  if (auto p = ctx->jumpStatement())
    return self(p);

  if (auto p = ctx->ifStatement())
    return self(p);

  if (auto p = ctx->whileStatement())
    return self(p);

  if (auto p = ctx->breakStatement())
    return self(p);

  ABORT();
}

CompoundStmt*
Ast2Asg::operator()(ast::CompoundStatementContext* ctx)
{
  auto ret = make<CompoundStmt>();

  if (auto p = ctx->blockItemList()) {
    Symtbl localDecls(self);

    for (auto&& i : p->blockItem()) {
      if (auto q = i->declaration()) {
        auto sub = make<DeclStmt>();
        sub->decls = self(q);
        ret->subs.push_back(sub);
      }

      else if (auto q = i->statement())
        ret->subs.push_back(self(q));

      else
        ABORT();
    }
  }

  return ret;
}

Stmt*
Ast2Asg::operator()(ast::ExpressionStatementContext* ctx)
{
  if (auto p = ctx->expression()) {
    auto ret = make<ExprStmt>();
    ret->expr = self(p);
    return ret;
  }

  return make<NullStmt>();
}

Stmt*
Ast2Asg::operator()(ast::JumpStatementContext* ctx)
{
  if (ctx->Return()) {
    auto ret = make<ReturnStmt>();
    ret->func = mCurrentFunc;
    if (auto p = ctx->expression())
      ret->expr = self(p);
    return ret;
  }

  ABORT();
}

// if语句-------------------------------------
IfStmt*
Ast2Asg::operator()(ast::IfStatementContext* ctx)
{
  auto ret = make<IfStmt>();
  auto children = ctx->children;

  //consiton-----------------------
  ast::Condition_highContext* temp_condition = dynamic_cast<ast::Condition_highContext*>(children[2]);
  ret->cond = self(temp_condition);
  
  //then---------------------------
  ast::CompoundStatementContext* temp_then = dynamic_cast<ast::CompoundStatementContext*>(children[4]);
  if(temp_then!=NULL)
    ret->then = self(temp_then);
  else{
    ast::BlockItemContext* temp_then_2 = dynamic_cast<ast::BlockItemContext*>(children[4]);
    ret->then = self(temp_then_2->statement());
  }
  
  //else----------------------------
  if(children.size()>=6){
    ast::CompoundStatementContext* temp_else = dynamic_cast<ast::CompoundStatementContext*>(children[6]);
    if(temp_else!=NULL)
      ret->else_ = self(temp_else); 
    else{
      ast::BlockItemContext* temp_else_2 = dynamic_cast<ast::BlockItemContext*>(children[6]);
      ret->else_ = self(temp_else_2->statement());  
    }
  }

  return ret;
}//---------------------------------------------


//while------------------------------------------
WhileStmt*
Ast2Asg::operator()(ast::WhileStatementContext* ctx)
{
  auto ret = make<WhileStmt>();
  auto children = ctx->children;

  //condition-----------------------
  if(auto p=ctx->condition_high())
    ret->cond = self(p);
  
  //body----------------------------
  if(auto p=ctx->compoundStatement())
    ret->body = self(p);
  else if(auto p=ctx->blockItem())
    ret->body = self(p->statement());  
    
  return ret;
}//---------------------------------------------



BreakStmt*
Ast2Asg::operator()(ast::BreakStatementContext* ctx)
{
  return make<BreakStmt>();
}

ContinueStmt*
Ast2Asg::operator()(ast::ContinueStatementContext* ctx)
{
  return make<ContinueStmt>();
}

//==============================================================================
// 声明
//==============================================================================

std::vector<Decl*>
Ast2Asg::operator()(ast::DeclarationContext* ctx)
{
  std::vector<Decl*> ret;

  auto specs = self(ctx->declarationSpecifiers());

  if (auto p = ctx->initDeclaratorList()) {
    for (auto&& j : p->initDeclarator())
      ret.push_back(self(j, specs));
  }

  // 如果 initDeclaratorList 为空则这行声明语句无意义
  return ret;
}

FunctionDecl*
Ast2Asg::operator()(ast::FunctionDefinitionContext* ctx)
{
  auto ret = make<FunctionDecl>();
  mCurrentFunc = ret;

  auto type = make<Type>();
  ret->type = type;

  auto sq = self(ctx->declarationSpecifiers());
  type->spec = sq.first, type->qual = sq.second;

  auto [texp, name] = self(ctx->directDeclarator(), nullptr);
  auto funcType = make<FunctionType>();
  funcType->sub = texp;
  type->texp = funcType;
  ret->name = std::move(name);

  Symtbl localDecls(self);

  if (auto p = ctx->declarationList()) {
    for (auto&& j : p->declaration())
      for (auto&& k : self(j))
        ret->params.push_back(k);
  }
  // 函数定义在签名之后就加入符号表，以允许递归调用
  (*mSymtbl)[ret->name] = ret;

  if (auto p = ctx->compoundStatement())
    ret->body = self(p);
  else
    ret->body = nullptr;
  return ret;
}

Decl*
Ast2Asg::operator()(ast::InitDeclaratorContext* ctx, SpecQual sq)
{
  auto [texp, name] = self(ctx->declarator(), nullptr);
  Decl* ret;

  if (auto funcType = texp->dcst<FunctionType>()) {
    auto fdecl = make<FunctionDecl>();
    auto type = make<Type>();
    fdecl->type = type;

    type->spec = sq.first;
    type->qual = sq.second;
    type->texp = funcType;

    fdecl->name = std::move(name);
    for (auto p : funcType->params) {
      auto paramDecl = make<VarDecl>();
      paramDecl->type = p;
      fdecl->params.push_back(paramDecl);
    }

    if (ctx->initializer())
      ABORT();
    fdecl->body = nullptr;

    ret = fdecl;
  }

  else {
    auto vdecl = make<VarDecl>();
    auto type = make<Type>();
    vdecl->type = type;

    type->spec = sq.first;
    type->qual = sq.second;
    type->texp = texp;
    vdecl->name = std::move(name);

    if (auto p = ctx->initializer())
      vdecl->init = self(p);
    else
      vdecl->init = nullptr;

    ret = vdecl;
  }

  // 这个实现允许符号重复定义，新定义会取代旧定义
  (*mSymtbl)[ret->name] = ret;
  return ret;
}

} // namespace asg