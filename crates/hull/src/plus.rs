use std::collections::{BTreeMap, BTreeSet};

use hir::{
    Db as HirDb,
    ast::function::{YulExpr, YulExprKind, YulLitKind, YulStmt, YulStmtKind},
    span::Span,
};

use crate::{
    emit::{EmitOptions, EmitOutput, emit_module},
    ir::{
        Alt, Arg, CodeBlock, Expr, ExprKind, Function, HullEvidence, Name, Object, PlaceExpr,
        PlaceProj, PlaceSpace, Program, Stmt, StmtKind, Ty, TyKind,
    },
};

const STORAGE_HASH2_HELPER: &str = "__solcore_storage_hash2";

/// Emit a [`specialize::MonoModule`] directly into the bounded `Hull+` slice.
pub fn emit_module_hull_plus<'db>(
    db: &'db dyn hir_ty::Db,
    module: &specialize::MonoModule<'db>,
    options: EmitOptions,
) -> EmitOutput<'db> {
    let mut output = emit_module(db, module, options);
    output.program = lift_program_to_hull_plus(db, &output.program);
    output
}

/// Check whether `program` contains any `Hull+` statements or expressions.
pub fn has_hull_plus(program: &Program<'_>) -> bool {
    program
        .functions
        .iter()
        .any(|f| stmts_have_hull_plus(&f.body))
        || program.objects.iter().any(object_has_hull_plus)
}

fn object_has_hull_plus(obj: &Object<'_>) -> bool {
    stmts_have_hull_plus(&obj.code.stmts)
        || obj
            .code
            .functions
            .iter()
            .any(|f| stmts_have_hull_plus(&f.body))
        || obj.inners.iter().any(object_has_hull_plus)
}

fn stmts_have_hull_plus(stmts: &[Stmt<'_>]) -> bool {
    stmts.iter().any(|s| match &s.kind {
        StmtKind::InitHeap
        | StmtKind::ParamMem { .. }
        | StmtKind::StorePlace { .. }
        | StmtKind::SplitPlace { .. }
        | StmtKind::JoinPlace { .. }
        | StmtKind::FreePlace { .. } => true,
        StmtKind::Assign { lhs, rhs } => expr_has_hull_plus(lhs) || expr_has_hull_plus(rhs),
        StmtKind::Expr(e) | StmtKind::Return(e) => expr_has_hull_plus(e),
        StmtKind::Block(ss) => stmts_have_hull_plus(ss),
        StmtKind::For {
            init,
            cond,
            post,
            body,
        } => {
            stmts_have_hull_plus(init)
                || expr_has_hull_plus(cond)
                || stmts_have_hull_plus(post)
                || stmts_have_hull_plus(body)
        }
        StmtKind::Match {
            scrutinee, alts, ..
        } => {
            expr_has_hull_plus(scrutinee) || alts.iter().any(|a| stmts_have_hull_plus(&a.body))
        }
        StmtKind::Let { .. }
        | StmtKind::Break
        | StmtKind::Continue
        | StmtKind::Assembly(_)
        | StmtKind::Revert(_)
        | StmtKind::Comment(_) => false,
    })
}

fn expr_has_hull_plus(expr: &Expr<'_>) -> bool {
    match &expr.kind {
        ExprKind::Alloc { .. } | ExprKind::LoadPlace { .. } => true,
        ExprKind::Pair(a, b) => expr_has_hull_plus(a) || expr_has_hull_plus(b),
        ExprKind::Fst(a)
        | ExprKind::Snd(a)
        | ExprKind::Inl { value: a, .. }
        | ExprKind::Inr { value: a, .. }
        | ExprKind::InK { value: a, .. } => expr_has_hull_plus(a),
        ExprKind::Call { args, .. } => args.iter().any(expr_has_hull_plus),
        ExprKind::If {
            cond,
            then_expr,
            else_expr,
            ..
        } => {
            expr_has_hull_plus(cond)
                || expr_has_hull_plus(then_expr)
                || expr_has_hull_plus(else_expr)
        }
        ExprKind::Word(_) | ExprKind::Bool(_) | ExprKind::Unit | ExprKind::Var(_) => false,
    }
}

/// Lift a Core Hull [`Program`] into the bounded `Hull+` slice:
/// - Free-memory pointer initializations (`mstore(0x40, 0x80)`) become [`StmtKind::InitHeap`].
/// - Arena bump allocations (`__solcore_alloc`, `allocate_memory`, or inline `mload(0x40)` /
///   `mstore(0x40, add(ptr, bytes))` including `__strlit_*` assembly blocks) become
///   [`ExprKind::Alloc`] with word-sized regions.
/// - Memory loads and stores rooted at tracked allocations (`mload(ptr + off)`,
///   `mstore(ptr + off, val)`) become [`ExprKind::LoadPlace`] and [`StmtKind::StorePlace`]
///   with [`PlaceProj::Field`] or [`PlaceProj::ByteOffset`] projections and [`HullEvidence`].
/// - Storage and transient storage accesses (`sload`, `sstore`, `tload`, `tstore`, and
///   `__solcore_storage_hash2` / `hash2` mapping slots) become [`ExprKind::LoadPlace`] and
///   [`StmtKind::StorePlace`] over [`PlaceSpace::Stor`] / [`PlaceSpace::TStor`].
pub fn lift_program_to_hull_plus<'db>(
    db: &'db dyn HirDb,
    program: &Program<'db>,
) -> Program<'db> {
    let mut ctx = LiftHelpers::analyze_functions(db, &program.functions);
    let functions = lift_function_list(db, &program.functions, &program.entry_points, &mut ctx);
    let objects = program
        .objects
        .iter()
        .map(|obj| lift_object(db, obj, &program.entry_points))
        .collect();
    Program {
        span: program.span,
        entry_points: program.entry_points.clone(),
        functions,
        objects,
    }
}

/// Erase all `Hull+` constructs in `program` back to legacy Core Hull:
/// - [`StmtKind::InitHeap`] -> `mstore(64, 128)`
/// - [`StmtKind::ParamMem`] -> erased (no-op at runtime)
/// - [`ExprKind::Alloc`] -> inline `mload(64)` + `mstore(64, add(ptr, bytes))`
/// - [`ExprKind::LoadPlace`] / [`StmtKind::StorePlace`] -> `mload`/`mstore`/`sload`/`sstore`/`tload`/`tstore`
///   with explicit byte/slot address arithmetic and `__solcore_storage_hash2` for map entries
/// - [`StmtKind::SplitPlace`] -> `binder := resolved_place_address`
/// - [`StmtKind::JoinPlace`] / [`StmtKind::FreePlace`] -> erased (lifetime-only markers)
pub fn erase_program<'db>(db: &'db dyn HirDb, program: &Program<'db>) -> Program<'db> {
    let _ = db;
    let mut eraser = Eraser::default();
    let functions = erase_function_list(&program.functions, program.span, &mut eraser);
    let objects = program
        .objects
        .iter()
        .map(|obj| erase_object(obj, &mut eraser))
        .collect();
    Program {
        span: program.span,
        entry_points: program.entry_points.clone(),
        functions,
        objects,
    }
}

#[derive(Default, Clone)]
struct LiftHelpers {
    identity_fns: BTreeSet<String>,
    mload_fns: BTreeSet<String>,
    mstore_fns: BTreeSet<String>,
    sload_fns: BTreeSet<String>,
    sstore_fns: BTreeSet<String>,
    add_fns: BTreeSet<String>,
    get_free_mem_fns: BTreeSet<String>,
    set_free_mem_fns: BTreeSet<String>,
    alloc_mem_fns: BTreeSet<String>,
    hash2_fns: BTreeSet<String>,
}

impl LiftHelpers {
    fn analyze_functions(db: &dyn HirDb, functions: &[Function<'_>]) -> Self {
        let mut helpers = Self::default();
        helpers.mload_fns.insert("mload".to_owned());
        helpers.mstore_fns.insert("mstore".to_owned());
        helpers.sload_fns.insert("sload".to_owned());
        helpers.sstore_fns.insert("sstore".to_owned());
        helpers.add_fns.insert("add".to_owned());
        helpers.add_fns.insert("primAddWord".to_owned());
        helpers.add_fns.insert("addWord".to_owned());
        helpers
            .get_free_mem_fns
            .insert("__solcore_free_memory_pointer".to_owned());
        helpers
            .set_free_mem_fns
            .insert("__solcore_set_free_memory_pointer".to_owned());
        helpers.alloc_mem_fns.insert("__solcore_alloc".to_owned());
        helpers.hash2_fns.insert(STORAGE_HASH2_HELPER.to_owned());

        loop {
            let mut changed = false;
            for func in functions {
                let name = func.name.as_str();
                if !helpers.identity_fns.contains(name)
                    && is_identity_word_fn(func, &helpers.identity_fns)
                {
                    helpers.identity_fns.insert(name.to_owned());
                    changed = true;
                }
                if !helpers.mload_fns.contains(name)
                    && is_unary_op_passthrough(
                        db,
                        func,
                        "mload",
                        &helpers.mload_fns,
                        &helpers.identity_fns,
                    )
                {
                    helpers.mload_fns.insert(name.to_owned());
                    changed = true;
                }
                if !helpers.mstore_fns.contains(name)
                    && is_binary_store_passthrough(
                        db,
                        func,
                        "mstore",
                        &helpers.mstore_fns,
                        &helpers.identity_fns,
                    )
                {
                    helpers.mstore_fns.insert(name.to_owned());
                    changed = true;
                }
                if !helpers.sload_fns.contains(name)
                    && is_unary_op_passthrough(
                        db,
                        func,
                        "sload",
                        &helpers.sload_fns,
                        &helpers.identity_fns,
                    )
                {
                    helpers.sload_fns.insert(name.to_owned());
                    changed = true;
                }
                if !helpers.sstore_fns.contains(name)
                    && is_binary_store_passthrough(
                        db,
                        func,
                        "sstore",
                        &helpers.sstore_fns,
                        &helpers.identity_fns,
                    )
                {
                    helpers.sstore_fns.insert(name.to_owned());
                    changed = true;
                }
                if !helpers.add_fns.contains(name)
                    && is_binary_word_passthrough(func, &helpers.add_fns, &helpers.identity_fns)
                {
                    helpers.add_fns.insert(name.to_owned());
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }

        for func in functions {
            let name = func.name.as_str();
            if name.starts_with("get_free_memory")
                || name.starts_with("std_get_free_memory")
                || is_get_free_mem_body(func, &helpers.mload_fns)
            {
                helpers.get_free_mem_fns.insert(name.to_owned());
            }
            if name.starts_with("set_free_memory")
                || name.starts_with("std_set_free_memory")
                || is_set_free_mem_body(func, &helpers.mstore_fns)
            {
                helpers.set_free_mem_fns.insert(name.to_owned());
            }
            if ((name.starts_with("hash2") || name.starts_with("std_hash2"))
                && func.args.len() == 2)
                || is_hash2_body(func)
            {
                helpers.hash2_fns.insert(name.to_owned());
            }
        }
        for func in functions {
            let name = func.name.as_str();
            if ((name.starts_with("allocate_memory") || name.starts_with("std_allocate_memory"))
                && !name.contains("allocate_zeroed_memory"))
                || is_alloc_mem_body(func, &helpers)
            {
                helpers.alloc_mem_fns.insert(name.to_owned());
            }
        }
        helpers
    }
}

fn is_word_like_ty(ty: &Ty<'_>) -> bool {
    matches!(ty.strip_named().kind, TyKind::Word)
}

fn is_unit_like_ty(ty: &Ty<'_>) -> bool {
    matches!(ty.strip_named().kind, TyKind::Unit)
}

fn is_identity_expr(expr: &Expr<'_>, param: &str, identity_fns: &BTreeSet<String>) -> bool {
    match &expr.kind {
        ExprKind::Var(v) => v.as_str() == param,
        ExprKind::Call { callee, args } if args.len() == 1 => {
            identity_fns.contains(callee.as_str())
                && is_identity_expr(&args[0], param, identity_fns)
        }
        _ => false,
    }
}

fn unwrap_identity_expr<'a, 'db>(
    mut expr: &'a Expr<'db>,
    identity_fns: &BTreeSet<String>,
) -> &'a Expr<'db> {
    while let ExprKind::Call { callee, args } = &expr.kind {
        if args.len() == 1 && identity_fns.contains(callee.as_str()) {
            expr = &args[0];
        } else {
            break;
        }
    }
    expr
}

fn is_identity_word_fn(func: &Function<'_>, identity_fns: &BTreeSet<String>) -> bool {
    let [arg] = func.args.as_slice() else {
        return false;
    };
    if !is_word_like_ty(&arg.ty) || !is_word_like_ty(&func.ret) {
        return false;
    }
    let param = arg.name.as_str();
    match func.body.as_slice() {
        [Stmt {
            kind: StmtKind::Return(ret_expr),
            ..
        }] => is_identity_expr(ret_expr, param, identity_fns),
        [
            Stmt {
                kind: StmtKind::Let { name: tmp1, .. },
                ..
            },
            Stmt {
                kind:
                    StmtKind::Assign {
                        lhs: Expr {
                            kind: ExprKind::Var(tmp2),
                            ..
                        },
                        rhs,
                    },
                ..
            },
            Stmt {
                kind:
                    StmtKind::Return(Expr {
                        kind: ExprKind::Var(tmp3),
                        ..
                    }),
                ..
            },
        ] => tmp1 == tmp2 && tmp2 == tmp3 && is_identity_expr(rhs, param, identity_fns),
        _ => false,
    }
}

fn is_unary_op_passthrough(
    db: &dyn HirDb,
    func: &Function<'_>,
    yul_op: &str,
    op_fns: &BTreeSet<String>,
    identity_fns: &BTreeSet<String>,
) -> bool {
    let [arg] = func.args.as_slice() else {
        return false;
    };
    if !is_word_like_ty(&arg.ty) || !is_word_like_ty(&func.ret) {
        return false;
    }
    let param = arg.name.as_str();
    let matches_call = |expr: &Expr<'_>| -> bool {
        let mut cur = expr;
        while let ExprKind::Call { callee, args } = &cur.kind {
            if args.len() != 1 {
                return false;
            }
            if op_fns.contains(callee.as_str()) {
                return is_identity_expr(&args[0], param, identity_fns);
            }
            if identity_fns.contains(callee.as_str()) {
                cur = &args[0];
            } else {
                return false;
            }
        }
        false
    };
    match func.body.as_slice() {
        [Stmt {
            kind: StmtKind::Return(ret_expr),
            ..
        }] => matches_call(ret_expr),
        [
            Stmt {
                kind: StmtKind::Let { name: tmp1, .. },
                ..
            },
            Stmt {
                kind:
                    StmtKind::Assign {
                        lhs: Expr {
                            kind: ExprKind::Var(tmp2),
                            ..
                        },
                        rhs,
                    },
                ..
            },
            Stmt {
                kind:
                    StmtKind::Return(Expr {
                        kind: ExprKind::Var(tmp3),
                        ..
                    }),
                ..
            },
        ] => tmp1 == tmp2 && tmp2 == tmp3 && matches_call(rhs),
        [
            Stmt {
                kind: StmtKind::Let { name: tmp1, .. },
                ..
            },
            Stmt {
                kind: StmtKind::Assembly(yul_stmts),
                ..
            },
            Stmt {
                kind:
                    StmtKind::Return(Expr {
                        kind: ExprKind::Var(tmp2),
                        ..
                    }),
                ..
            },
        ] if tmp1 == tmp2 && yul_stmts.len() == 1 => {
            if let YulStmtKind::Assign { names, value } = &yul_stmts[0].kind
                && names.len() == 1
                && (*names[0].atom()).text(db) == tmp1.as_str()
                && let YulExprKind::Call { name, args } = &value.kind
                && (*name.atom()).text(db) == yul_op
                && args.len() == 1
                && let YulExprKind::Ident(id) = &args[0].kind
            {
                (*id.atom()).text(db) == param
            } else {
                false
            }
        }
        _ => false,
    }
}

fn is_binary_store_passthrough(
    db: &dyn HirDb,
    func: &Function<'_>,
    yul_op: &str,
    store_fns: &BTreeSet<String>,
    identity_fns: &BTreeSet<String>,
) -> bool {
    let [arg0, arg1] = func.args.as_slice() else {
        return false;
    };
    if !is_word_like_ty(&arg0.ty) || !is_word_like_ty(&arg1.ty) || !is_unit_like_ty(&func.ret) {
        return false;
    }
    let p0 = arg0.name.as_str();
    let p1 = arg1.name.as_str();
    let matches_call = |expr: &Expr<'_>| -> bool {
        if let ExprKind::Call { callee, args } = &expr.kind
            && args.len() == 2
            && store_fns.contains(callee.as_str())
        {
            is_identity_expr(&args[0], p0, identity_fns)
                && is_identity_expr(&args[1], p1, identity_fns)
        } else {
            false
        }
    };
    match func.body.as_slice() {
        [Stmt {
            kind: StmtKind::Expr(e),
            ..
        }]
        | [
            Stmt {
                kind: StmtKind::Expr(e),
                ..
            },
            Stmt {
                kind: StmtKind::Return(_),
                ..
            },
        ] => matches_call(e),
        [Stmt {
            kind: StmtKind::Assembly(yul_stmts),
            ..
        }] if yul_stmts.len() == 1 => {
            if let YulStmtKind::Expr(YulExpr {
                kind: YulExprKind::Call { name, args },
                ..
            }) = &yul_stmts[0].kind
                && (*name.atom()).text(db) == yul_op
                && args.len() == 2
                && let (YulExprKind::Ident(i0), YulExprKind::Ident(i1)) =
                    (&args[0].kind, &args[1].kind)
            {
                (*i0.atom()).text(db) == p0 && (*i1.atom()).text(db) == p1
            } else {
                false
            }
        }
        _ => false,
    }
}

fn is_binary_word_passthrough(
    func: &Function<'_>,
    op_fns: &BTreeSet<String>,
    identity_fns: &BTreeSet<String>,
) -> bool {
    let [arg0, arg1] = func.args.as_slice() else {
        return false;
    };
    if !is_word_like_ty(&arg0.ty) || !is_word_like_ty(&arg1.ty) || !is_word_like_ty(&func.ret) {
        return false;
    }
    let p0 = arg0.name.as_str();
    let p1 = arg1.name.as_str();
    let [Stmt {
        kind: StmtKind::Return(ret_expr),
        ..
    }] = func.body.as_slice()
    else {
        return false;
    };
    let mut cur = ret_expr;
    while let ExprKind::Call { callee, args } = &cur.kind {
        if args.len() == 2 && op_fns.contains(callee.as_str()) {
            return is_identity_expr(&args[0], p0, identity_fns)
                && is_identity_expr(&args[1], p1, identity_fns);
        }
        if args.len() == 1 && identity_fns.contains(callee.as_str()) {
            cur = &args[0];
        } else {
            return false;
        }
    }
    false
}

fn is_zero_or_64(text: &str, expected: u64) -> bool {
    parse_u64_lit(text) == Some(expected)
}

pub fn parse_u64_lit(text: &str) -> Option<u64> {
    let trimmed = text.trim();
    if let Some(hex) = trimmed
        .strip_prefix("0x")
        .or_else(|| trimmed.strip_prefix("0X"))
    {
        u64::from_str_radix(hex, 16).ok()
    } else {
        trimmed.parse::<u64>().ok()
    }
}

fn is_get_free_mem_body(func: &Function<'_>, mload_fns: &BTreeSet<String>) -> bool {
    if !func.args.is_empty() {
        return false;
    }
    match func.body.as_slice() {
        [Stmt {
            kind: StmtKind::Return(expr),
            ..
        }] => is_mload_0x40(expr, mload_fns),
        _ => false,
    }
}

fn is_set_free_mem_body(func: &Function<'_>, mstore_fns: &BTreeSet<String>) -> bool {
    let [arg] = func.args.as_slice() else {
        return false;
    };
    let param = arg.name.as_str();
    match func.body.as_slice() {
        [Stmt {
            kind: StmtKind::Expr(expr),
            ..
        }]
        | [
            Stmt {
                kind: StmtKind::Expr(expr),
                ..
            },
            Stmt {
                kind: StmtKind::Return(_),
                ..
            },
        ] => is_mstore_0x40_var(expr, param, mstore_fns),
        _ => false,
    }
}

fn is_hash2_body(func: &Function<'_>) -> bool {
    func.args.len() == 2 && func.name.as_str() == STORAGE_HASH2_HELPER
}

fn is_alloc_mem_body(func: &Function<'_>, helpers: &LiftHelpers) -> bool {
    func.args.len() == 1
        && func
            .body
            .iter()
            .any(|s| stmt_calls_any(s, &helpers.get_free_mem_fns))
        && func
            .body
            .iter()
            .any(|s| stmt_calls_any(s, &helpers.set_free_mem_fns))
}

fn stmt_calls_any(stmt: &Stmt<'_>, names: &BTreeSet<String>) -> bool {
    match &stmt.kind {
        StmtKind::Assign { lhs, rhs } => {
            expr_calls_any(lhs, names) || expr_calls_any(rhs, names)
        }
        StmtKind::Expr(rhs) | StmtKind::Return(rhs) => expr_calls_any(rhs, names),
        StmtKind::Block(ss) => ss.iter().any(|s| stmt_calls_any(s, names)),
        StmtKind::For {
            init,
            cond,
            post,
            body,
        } => {
            init.iter().any(|s| stmt_calls_any(s, names))
                || expr_calls_any(cond, names)
                || post.iter().any(|s| stmt_calls_any(s, names))
                || body.iter().any(|s| stmt_calls_any(s, names))
        }
        StmtKind::Match {
            scrutinee, alts, ..
        } => {
            expr_calls_any(scrutinee, names)
                || alts
                    .iter()
                    .any(|a| a.body.iter().any(|s| stmt_calls_any(s, names)))
        }
        StmtKind::StorePlace { place, value, .. } => {
            place_calls_any(place, names) || expr_calls_any(value, names)
        }
        StmtKind::SplitPlace { place, .. }
        | StmtKind::JoinPlace { place }
        | StmtKind::FreePlace { place } => place_calls_any(place, names),
        _ => false,
    }
}

fn place_calls_any(place: &PlaceExpr<'_>, names: &BTreeSet<String>) -> bool {
    expr_calls_any(&place.base, names)
        || place.projections.iter().any(|p| match p {
            PlaceProj::Field { .. } | PlaceProj::Lane { .. } => false,
            PlaceProj::Index {
                index, bound_elems, ..
            } => {
                expr_calls_any(index, names)
                    || bound_elems.as_ref().is_some_and(|b| expr_calls_any(b, names))
            }
            PlaceProj::Entry { key, .. } => expr_calls_any(key, names),
            PlaceProj::ByteOffset { byte_offset, .. } => expr_calls_any(byte_offset, names),
        })
}

fn expr_calls_any(expr: &Expr<'_>, names: &BTreeSet<String>) -> bool {
    match &expr.kind {
        ExprKind::Call { callee, args } => {
            names.contains(callee.as_str()) || args.iter().any(|a| expr_calls_any(a, names))
        }
        ExprKind::Alloc { words } => expr_calls_any(words, names),
        ExprKind::LoadPlace { place, .. } => place_calls_any(place, names),
        ExprKind::Pair(a, b) => expr_calls_any(a, names) || expr_calls_any(b, names),
        ExprKind::Fst(a)
        | ExprKind::Snd(a)
        | ExprKind::Inl { value: a, .. }
        | ExprKind::Inr { value: a, .. }
        | ExprKind::InK { value: a, .. } => expr_calls_any(a, names),
        ExprKind::If {
            cond,
            then_expr,
            else_expr,
            ..
        } => {
            expr_calls_any(cond, names)
                || expr_calls_any(then_expr, names)
                || expr_calls_any(else_expr, names)
        }
        _ => false,
    }
}

fn is_mload_0x40(expr: &Expr<'_>, mload_fns: &BTreeSet<String>) -> bool {
    matches!(
        &expr.kind,
        ExprKind::Call { callee, args }
            if mload_fns.contains(callee.as_str())
                && matches!(args.as_slice(), [Expr { kind: ExprKind::Word(w), .. }] if is_zero_or_64(w, 0x40))
    )
}

fn is_mstore_0x40_var(expr: &Expr<'_>, var_name: &str, mstore_fns: &BTreeSet<String>) -> bool {
    matches!(
        &expr.kind,
        ExprKind::Call { callee, args }
            if mstore_fns.contains(callee.as_str())
                && matches!(
                    args.as_slice(),
                    [
                        Expr { kind: ExprKind::Word(w), .. },
                        Expr { kind: ExprKind::Var(v), .. }
                    ] if is_zero_or_64(w, 0x40) && v.as_str() == var_name
                )
    )
}

fn lift_object<'db>(
    db: &'db dyn HirDb,
    obj: &Object<'db>,
    entry_points: &[Name],
) -> Object<'db> {
    let mut obj_entries: Vec<Name> = entry_points.to_vec();
    for s in &obj.code.stmts {
        if let StmtKind::Expr(Expr {
            kind: ExprKind::Call { callee, args },
            ..
        }) = &s.kind
            && args.is_empty()
            && !obj_entries.contains(callee)
        {
            obj_entries.push(callee.clone());
        }
    }
    let mut ctx = LiftHelpers::analyze_functions(db, &obj.code.functions);
    let mut code_stmts = LiftState::new(db, &ctx, false).lift_stmts(&obj.code.stmts);
    if !code_stmts
        .iter()
        .any(|s| matches!(s.kind, StmtKind::InitHeap))
        && code_stmts_have_init_heap_yul(db, &code_stmts)
    {
        code_stmts = replace_init_heap_yul(db, code_stmts);
    }
    let functions = lift_function_list(db, &obj.code.functions, &obj_entries, &mut ctx);
    let inners = obj
        .inners
        .iter()
        .map(|inner| lift_object(db, inner, entry_points))
        .collect();
    Object {
        span: obj.span,
        name: obj.name.clone(),
        code: CodeBlock {
            span: obj.code.span,
            stmts: code_stmts,
            functions,
        },
        inners,
    }
}

fn lift_function_list<'db>(
    db: &'db dyn HirDb,
    functions: &[Function<'db>],
    entry_points: &[Name],
    ctx: &mut LiftHelpers,
) -> Vec<Function<'db>> {
    let mut lifted: Vec<Function<'db>> = functions
        .iter()
        .map(|func| {
            let is_entry =
                func.name.as_str() == "main" || entry_points.iter().any(|e| e == &func.name);
            lift_function(db, func, ctx, is_entry)
        })
        .collect();

    let group_has_alloc = lifted.iter().any(|f| stmts_have_alloc(&f.body));
    if group_has_alloc {
        for func in &mut lifted {
            let is_entry =
                func.name.as_str() == "main" || entry_points.iter().any(|e| e == &func.name);
            if is_entry && !stmts_have_init_heap(&func.body) {
                func.body.insert(
                    0,
                    Stmt {
                        span: func.span,
                        kind: StmtKind::InitHeap,
                    },
                );
            }
        }
    }

    let hash2_still_called = lifted.iter().any(|f| {
        f.name.as_str() != STORAGE_HASH2_HELPER
            && f.body.iter().any(|s| stmt_calls_any(s, &ctx.hash2_fns))
    });
    if !hash2_still_called {
        lifted.retain(|f| f.name.as_str() != STORAGE_HASH2_HELPER);
    }

    lifted
}

fn stmts_have_init_heap(stmts: &[Stmt<'_>]) -> bool {
    stmts.iter().any(|s| match &s.kind {
        StmtKind::InitHeap => true,
        StmtKind::Block(ss) => stmts_have_init_heap(ss),
        StmtKind::For {
            init, post, body, ..
        } => stmts_have_init_heap(init) || stmts_have_init_heap(post) || stmts_have_init_heap(body),
        StmtKind::Match { alts, .. } => alts.iter().any(|a| stmts_have_init_heap(&a.body)),
        _ => false,
    })
}

fn stmts_have_alloc(stmts: &[Stmt<'_>]) -> bool {
    stmts.iter().any(|s| match &s.kind {
        StmtKind::Assign { lhs, rhs } => expr_has_alloc(lhs) || expr_has_alloc(rhs),
        StmtKind::Expr(e) | StmtKind::Return(e) => expr_has_alloc(e),
        StmtKind::Block(ss) => stmts_have_alloc(ss),
        StmtKind::For {
            init,
            cond,
            post,
            body,
        } => {
            stmts_have_alloc(init)
                || expr_has_alloc(cond)
                || stmts_have_alloc(post)
                || stmts_have_alloc(body)
        }
        StmtKind::Match {
            scrutinee, alts, ..
        } => expr_has_alloc(scrutinee) || alts.iter().any(|a| stmts_have_alloc(&a.body)),
        _ => false,
    })
}

fn expr_has_alloc(expr: &Expr<'_>) -> bool {
    match &expr.kind {
        ExprKind::Alloc { .. } => true,
        ExprKind::LoadPlace { place, .. } => expr_has_alloc(&place.base),
        ExprKind::Pair(a, b) => expr_has_alloc(a) || expr_has_alloc(b),
        ExprKind::Fst(a)
        | ExprKind::Snd(a)
        | ExprKind::Inl { value: a, .. }
        | ExprKind::Inr { value: a, .. }
        | ExprKind::InK { value: a, .. } => expr_has_alloc(a),
        ExprKind::Call { args, .. } => args.iter().any(expr_has_alloc),
        ExprKind::If {
            cond,
            then_expr,
            else_expr,
            ..
        } => expr_has_alloc(cond) || expr_has_alloc(then_expr) || expr_has_alloc(else_expr),
        ExprKind::Word(_) | ExprKind::Bool(_) | ExprKind::Unit | ExprKind::Var(_) => false,
    }
}

fn lift_function<'db>(
    db: &'db dyn HirDb,
    func: &Function<'db>,
    ctx: &LiftHelpers,
    is_entry: bool,
) -> Function<'db> {
    if ctx.get_free_mem_fns.contains(func.name.as_str())
        || ctx.set_free_mem_fns.contains(func.name.as_str())
        || ctx.alloc_mem_fns.contains(func.name.as_str())
        || ctx.hash2_fns.contains(func.name.as_str())
        || ctx.mload_fns.contains(func.name.as_str())
        || ctx.mstore_fns.contains(func.name.as_str())
        || ctx.sload_fns.contains(func.name.as_str())
        || ctx.sstore_fns.contains(func.name.as_str())
    {
        return func.clone();
    }

    if let Some(lifted_strlit) = try_lift_strlit_function(db, func) {
        return lifted_strlit;
    }

    let mut state = LiftState::new(db, ctx, is_entry);
    let mut body = state.lift_stmts(&func.body);

    let top_level_allocs: Vec<(Name, usize)> = body
        .iter()
        .filter_map(|s| {
            if let StmtKind::Assign {
                lhs: Expr {
                    kind: ExprKind::Var(name),
                    ..
                },
                rhs: Expr {
                    kind: ExprKind::Alloc { .. },
                    ..
                },
            } = &s.kind
            {
                state
                    .alloc_vars
                    .get(name.as_str())
                    .copied()
                    .map(|sz| (name.clone(), sz))
            } else {
                None
            }
        })
        .collect();

    if !top_level_allocs.is_empty() {
        let ret_mentions = match body.last() {
            Some(Stmt {
                kind: StmtKind::Return(ret_expr),
                ..
            }) => Some(ret_expr.clone()),
            _ => None,
        };
        let insert_idx = if ret_mentions.is_some() {
            body.len() - 1
        } else {
            body.len()
        };
        let mut frees = Vec::new();
        for (name, sz_words) in top_level_allocs.into_iter().rev() {
            if ret_mentions
                .as_ref()
                .is_some_and(|e| expr_mentions_var(e, name.as_str()))
            {
                continue;
            }
            frees.push(Stmt {
                span: func.span,
                kind: StmtKind::FreePlace {
                    place: PlaceExpr::mem(
                        Expr::var(func.span, name, Ty::word(func.span)),
                        sz_words,
                    ),
                },
            });
        }
        body.splice(insert_idx..insert_idx, frees);
    }

    if is_entry
        && state.used_mem_alloc
        && !body.iter().any(|s| matches!(s.kind, StmtKind::InitHeap))
    {
        body.insert(
            0,
            Stmt {
                span: func.span,
                kind: StmtKind::InitHeap,
            },
        );
    }

    Function {
        span: func.span,
        name: func.name.clone(),
        args: func.args.clone(),
        ret: func.ret.clone(),
        body,
    }
}

fn expr_mentions_var(expr: &Expr<'_>, target: &str) -> bool {
    match &expr.kind {
        ExprKind::Var(v) => v.as_str() == target,
        ExprKind::Alloc { words } => expr_mentions_var(words, target),
        ExprKind::LoadPlace { place, .. } => {
            expr_mentions_var(&place.base, target)
                || place.projections.iter().any(|p| match p {
                    PlaceProj::Field { .. } | PlaceProj::Lane { .. } => false,
                    PlaceProj::Index {
                        index, bound_elems, ..
                    } => {
                        expr_mentions_var(index, target)
                            || bound_elems
                                .as_ref()
                                .is_some_and(|b| expr_mentions_var(b, target))
                    }
                    PlaceProj::Entry { key, .. } => expr_mentions_var(key, target),
                    PlaceProj::ByteOffset { byte_offset, .. } => {
                        expr_mentions_var(byte_offset, target)
                    }
                })
        }
        ExprKind::Pair(a, b) => expr_mentions_var(a, target) || expr_mentions_var(b, target),
        ExprKind::Fst(a)
        | ExprKind::Snd(a)
        | ExprKind::Inl { value: a, .. }
        | ExprKind::Inr { value: a, .. }
        | ExprKind::InK { value: a, .. } => expr_mentions_var(a, target),
        ExprKind::Call { args, .. } => args.iter().any(|a| expr_mentions_var(a, target)),
        ExprKind::If {
            cond,
            then_expr,
            else_expr,
            ..
        } => {
            expr_mentions_var(cond, target)
                || expr_mentions_var(then_expr, target)
                || expr_mentions_var(else_expr, target)
        }
        ExprKind::Word(_) | ExprKind::Bool(_) | ExprKind::Unit => false,
    }
}

fn try_lift_strlit_function<'db>(
    db: &'db dyn HirDb,
    func: &Function<'db>,
) -> Option<Function<'db>> {
    if !func.args.is_empty() {
        return None;
    }
    let [
        Stmt {
            kind: StmtKind::Let { name: ptr_name, ty },
            span: let_span,
        },
        Stmt {
            kind: StmtKind::Assembly(yul_stmts),
            span: asm_span,
        },
        Stmt {
            kind: StmtKind::Return(ret_expr),
            span: ret_span,
        },
    ] = func.body.as_slice()
    else {
        return None;
    };
    let ExprKind::Var(ret_var) = &ret_expr.kind else {
        return None;
    };
    if ret_var != ptr_name || yul_stmts.len() < 2 {
        return None;
    }
    let first = yul_stmts.first()?;
    if !is_yul_assign_mload_0x40(db, first, ptr_name.as_str()) {
        return None;
    }
    let last = yul_stmts.last()?;
    let total_bytes = yul_mstore_0x40_bump_bytes(db, last, ptr_name.as_str())?;
    if total_bytes == 0 || total_bytes % 32 != 0 {
        return None;
    }
    let total_words = (total_bytes / 32) as usize;
    let mut stores = Vec::new();
    for ystmt in &yul_stmts[1..yul_stmts.len() - 1] {
        let (word_offset, val_str) = yul_mstore_const_field(db, ystmt, ptr_name.as_str())?;
        if word_offset >= total_words {
            return None;
        }
        let place = PlaceExpr::mem(
            Expr::var(*asm_span, ptr_name.clone(), Ty::word(*asm_span)),
            total_words,
        )
        .field(word_offset, 1);
        stores.push(Stmt {
            span: ystmt.span,
            kind: StmtKind::StorePlace {
                place,
                value: Expr::word(ystmt.span, val_str),
                evidence: HullEvidence::Checked,
            },
        });
    }

    let mut body = Vec::with_capacity(2 + stores.len() + 1);
    body.push(Stmt {
        span: *let_span,
        kind: StmtKind::Let {
            name: ptr_name.clone(),
            ty: ty.clone(),
        },
    });
    body.push(Stmt {
        span: *asm_span,
        kind: StmtKind::Assign {
            lhs: Expr::var(*asm_span, ptr_name.clone(), ty.clone()),
            rhs: Expr {
                span: *asm_span,
                ty: Ty::word(*asm_span),
                kind: ExprKind::Alloc {
                    words: Box::new(Expr::word(*asm_span, total_words.to_string())),
                },
            },
        },
    });
    body.extend(stores);
    body.push(Stmt {
        span: *ret_span,
        kind: StmtKind::Return(ret_expr.clone()),
    });

    Some(Function {
        span: func.span,
        name: func.name.clone(),
        args: func.args.clone(),
        ret: func.ret.clone(),
        body,
    })
}

fn is_yul_assign_mload_0x40(db: &dyn HirDb, stmt: &YulStmt<'_>, expected_var: &str) -> bool {
    let YulStmtKind::Assign { names, value } = &stmt.kind else {
        return false;
    };
    if names.len() != 1 || (*names[0].atom()).text(db) != expected_var {
        return false;
    }
    let YulExprKind::Call { name, args } = &value.kind else {
        return false;
    };
    (*name.atom()).text(db) == "mload"
        && matches!(args.as_slice(), [arg] if is_yul_lit_u64(arg, 0x40))
}

fn yul_mstore_0x40_bump_bytes(
    db: &dyn HirDb,
    stmt: &YulStmt<'_>,
    expected_var: &str,
) -> Option<u64> {
    let YulStmtKind::Expr(YulExpr {
        kind: YulExprKind::Call { name, args },
        ..
    }) = &stmt.kind
    else {
        return None;
    };
    if (*name.atom()).text(db) != "mstore" || args.len() != 2 {
        return None;
    }
    if !is_yul_lit_u64(&args[0], 0x40) {
        return None;
    }
    let YulExprKind::Call {
        name: add_name,
        args: add_args,
    } = &args[1].kind
    else {
        return None;
    };
    if (*add_name.atom()).text(db) != "add" || add_args.len() != 2 {
        return None;
    }
    let YulExprKind::Ident(v) = &add_args[0].kind else {
        return None;
    };
    if (*v.atom()).text(db) != expected_var {
        return None;
    }
    let YulExprKind::Lit(YulLitKind::Number(sz) | YulLitKind::Hex(sz)) = &add_args[1].kind else {
        return None;
    };
    parse_u64_lit(sz)
}

fn yul_mstore_const_field(
    db: &dyn HirDb,
    stmt: &YulStmt<'_>,
    expected_var: &str,
) -> Option<(usize, String)> {
    let YulStmtKind::Expr(YulExpr {
        kind: YulExprKind::Call { name, args },
        ..
    }) = &stmt.kind
    else {
        return None;
    };
    if (*name.atom()).text(db) != "mstore" || args.len() != 2 {
        return None;
    }
    let word_offset = match &args[0].kind {
        YulExprKind::Ident(v) if (*v.atom()).text(db) == expected_var => 0usize,
        YulExprKind::Call {
            name: add_name,
            args: add_args,
        } if (*add_name.atom()).text(db) == "add" && add_args.len() == 2 => {
            let YulExprKind::Ident(v) = &add_args[0].kind else {
                return None;
            };
            if (*v.atom()).text(db) != expected_var {
                return None;
            }
            let YulExprKind::Lit(YulLitKind::Number(off) | YulLitKind::Hex(off)) =
                &add_args[1].kind
            else {
                return None;
            };
            let byte_off = parse_u64_lit(off)?;
            if byte_off % 32 != 0 {
                return None;
            }
            (byte_off / 32) as usize
        }
        _ => return None,
    };
    let YulExprKind::Lit(YulLitKind::Number(val) | YulLitKind::Hex(val)) = &args[1].kind else {
        return None;
    };
    Some((word_offset, val.clone()))
}

struct LiftState<'a, 'db> {
    db: &'db dyn HirDb,
    ctx: &'a LiftHelpers,
    is_entry: bool,
    used_mem_alloc: bool,
    alloc_vars: BTreeMap<String, usize>,
    hash2_vars: BTreeMap<String, (Expr<'db>, Expr<'db>)>,
}

impl<'a, 'db> LiftState<'a, 'db> {
    fn new(db: &'db dyn HirDb, ctx: &'a LiftHelpers, is_entry: bool) -> Self {
        Self {
            db,
            ctx,
            is_entry,
            used_mem_alloc: false,
            alloc_vars: BTreeMap::new(),
            hash2_vars: BTreeMap::new(),
        }
    }

    fn lift_stmts(&mut self, stmts: &[Stmt<'db>]) -> Vec<Stmt<'db>> {
        let mut out = Vec::with_capacity(stmts.len());
        let mut i = 0;
        while i < stmts.len() {
            if i + 1 < stmts.len()
                && let Some((ptr_lhs, words_expr, size_words_opt)) =
                    self.try_match_two_stmt_alloc(&stmts[i], &stmts[i + 1])
            {
                let span = stmts[i].span;
                if let ExprKind::Var(name) = &ptr_lhs.kind
                    && let Some(sz_words) = size_words_opt
                {
                    self.alloc_vars.insert(name.as_str().to_owned(), sz_words);
                }
                self.used_mem_alloc = true;
                out.push(Stmt {
                    span,
                    kind: StmtKind::Assign {
                        lhs: ptr_lhs,
                        rhs: Expr {
                            span,
                            ty: Ty::word(span),
                            kind: ExprKind::Alloc {
                                words: Box::new(words_expr),
                            },
                        },
                    },
                });
                i += 2;
                continue;
            }

            if i + 4 < stmts.len()
                && let Some(lifted_seq) = self.try_match_storage_index_store_seq(&stmts[i..i + 5])
            {
                out.extend(lifted_seq);
                i += 5;
                continue;
            }

            if i + 2 < stmts.len()
                && let Some(lifted_seq) = self.try_match_three_stmt_storage_store(&stmts[i..i + 3])
            {
                out.extend(lifted_seq);
                i += 3;
                continue;
            }

            let lifted = self.lift_stmt(&stmts[i]);
            out.extend(lifted);
            i += 1;
        }
        let _ = self.is_entry;
        out
    }

    fn try_match_two_stmt_alloc(
        &mut self,
        s1: &Stmt<'db>,
        s2: &Stmt<'db>,
    ) -> Option<(Expr<'db>, Expr<'db>, Option<usize>)> {
        let StmtKind::Assign { lhs, rhs } = &s1.kind else {
            return None;
        };
        let ExprKind::Var(ptr_name) = &lhs.kind else {
            return None;
        };
        let is_free_read = is_mload_0x40(rhs, &self.ctx.mload_fns)
            || matches!(
                &rhs.kind,
                ExprKind::Call { callee, args }
                    if self.ctx.get_free_mem_fns.contains(callee.as_str()) && args.is_empty()
            );
        if !is_free_read {
            return None;
        }
        let StmtKind::Expr(e2) = &s2.kind else {
            return None;
        };
        let bump_arg = match &e2.kind {
            ExprKind::Call { callee, args }
                if self.ctx.mstore_fns.contains(callee.as_str())
                    && args.len() == 2
                    && matches!(&args[0].kind, ExprKind::Word(w) if is_zero_or_64(w, 0x40)) =>
            {
                &args[1]
            }
            ExprKind::Call { callee, args }
                if self.ctx.set_free_mem_fns.contains(callee.as_str()) && args.len() == 1 =>
            {
                &args[0]
            }
            _ => return None,
        };
        let ExprKind::Call {
            callee: add_callee,
            args: add_args,
        } = &bump_arg.kind
        else {
            return None;
        };
        if !self.ctx.add_fns.contains(add_callee.as_str()) || add_args.len() != 2 {
            return None;
        }
        let bytes_expr = match (&add_args[0].kind, &add_args[1].kind) {
            (ExprKind::Var(v), _) if v == ptr_name => &add_args[1],
            (_, ExprKind::Var(v)) if v == ptr_name => &add_args[0],
            _ => return None,
        };
        let (words_expr, sz_words_opt) = self.bytes_expr_to_words(bytes_expr.clone())?;
        Some((lhs.clone(), words_expr, sz_words_opt))
    }

    fn try_match_storage_index_store_seq(&mut self, window: &[Stmt<'db>]) -> Option<Vec<Stmt<'db>>> {
        let [s0, s1, s2, s3, s4] = window else {
            return None;
        };
        let StmtKind::Let { name: slot_var, .. } = &s0.kind else {
            return None;
        };
        let StmtKind::Assign {
            lhs: Expr {
                kind: ExprKind::Var(lhs_slot),
                ..
            },
            rhs: hash_rhs,
        } = &s1.kind
        else {
            return None;
        };
        if lhs_slot != slot_var {
            return None;
        }
        let (w0, w1) = self.match_hash2_call(hash_rhs)?;
        let StmtKind::Let {
            name: val_var,
            ty: val_ty,
        } = &s2.kind
        else {
            return None;
        };
        let StmtKind::Assign {
            lhs: Expr {
                kind: ExprKind::Var(lhs_val),
                ..
            },
            rhs: val_rhs,
        } = &s3.kind
        else {
            return None;
        };
        if lhs_val != val_var {
            return None;
        }
        let StmtKind::Expr(Expr {
            kind: ExprKind::Call { callee, args },
            ..
        }) = &s4.kind
        else {
            return None;
        };
        if !self.ctx.sstore_fns.contains(callee.as_str()) || args.len() != 2 {
            return None;
        }
        if !matches!(&args[0].kind, ExprKind::Var(v) if v == slot_var)
            || !matches!(&args[1].kind, ExprKind::Var(v) if v == val_var)
        {
            return None;
        }

        let w0 = self.lift_expr(w0);
        let w1 = self.lift_expr(w1);
        self.hash2_vars
            .insert(slot_var.as_str().to_owned(), (w0.clone(), w1.clone()));
        let lifted_val_rhs = self.lift_expr(val_rhs.clone());
        self.hash2_vars.remove(slot_var.as_str());

        let place = PlaceExpr::stor(w1, 1).entry(w0, 1);
        Some(vec![
            Stmt {
                span: s2.span,
                kind: StmtKind::Let {
                    name: val_var.clone(),
                    ty: val_ty.clone(),
                },
            },
            Stmt {
                span: s3.span,
                kind: StmtKind::Assign {
                    lhs: Expr::var(s3.span, val_var.clone(), val_ty.clone()),
                    rhs: lifted_val_rhs,
                },
            },
            Stmt {
                span: s4.span,
                kind: StmtKind::StorePlace {
                    place,
                    value: Expr::var(s4.span, val_var.clone(), Ty::word(s4.span)),
                    evidence: HullEvidence::Checked,
                },
            },
        ])
    }

    fn try_match_three_stmt_storage_store(
        &mut self,
        window: &[Stmt<'db>],
    ) -> Option<Vec<Stmt<'db>>> {
        let [s0, s1, s2] = window else {
            return None;
        };
        let StmtKind::Let { name: slot_var, .. } = &s0.kind else {
            return None;
        };
        let StmtKind::Assign {
            lhs: Expr {
                kind: ExprKind::Var(lhs_slot),
                ..
            },
            rhs: slot_rhs,
        } = &s1.kind
        else {
            return None;
        };
        if lhs_slot != slot_var {
            return None;
        }
        let StmtKind::Expr(Expr {
            kind: ExprKind::Call { callee, args },
            ..
        }) = &s2.kind
        else {
            return None;
        };
        if !self.ctx.sstore_fns.contains(callee.as_str()) || args.len() != 2 {
            return None;
        }
        if !is_identity_expr(&args[0], slot_var.as_str(), &self.ctx.identity_fns)
            || expr_mentions_var(&args[1], slot_var.as_str())
        {
            return None;
        }
        let place = self.storage_place_from_slot(slot_rhs.clone());
        let value = self.lift_expr(args[1].clone());
        Some(vec![Stmt {
            span: s2.span,
            kind: StmtKind::StorePlace {
                place,
                value,
                evidence: HullEvidence::Checked,
            },
        }])
    }

    fn match_hash2_call(&self, expr: &Expr<'db>) -> Option<(Expr<'db>, Expr<'db>)> {
        let ExprKind::Call { callee, args } = &expr.kind else {
            return None;
        };
        if self.ctx.hash2_fns.contains(callee.as_str()) && args.len() == 2 {
            Some((args[0].clone(), args[1].clone()))
        } else {
            None
        }
    }

    fn bytes_expr_to_words(&mut self, bytes_expr: Expr<'db>) -> Option<(Expr<'db>, Option<usize>)> {
        let lifted = self.lift_expr(bytes_expr);
        if let ExprKind::Word(text) = &lifted.kind
            && let Some(bytes) = parse_u64_lit(text)
            && bytes > 0
            && bytes % 32 == 0
        {
            let words = (bytes / 32) as usize;
            return Some((Expr::word(lifted.span, words.to_string()), Some(words)));
        }
        if let ExprKind::Call { callee, args } = &lifted.kind {
            if callee.as_str() == "mul" && args.len() == 2 {
                if matches!(&args[0].kind, ExprKind::Word(w) if parse_u64_lit(w) == Some(32)) {
                    return Some((args[1].clone(), None));
                }
                if matches!(&args[1].kind, ExprKind::Word(w) if parse_u64_lit(w) == Some(32)) {
                    return Some((args[0].clone(), None));
                }
            }
            if callee.as_str() == "shl"
                && args.len() == 2
                && matches!(&args[0].kind, ExprKind::Word(w) if parse_u64_lit(w) == Some(5))
            {
                return Some((args[1].clone(), None));
            }
        }
        None
    }

    fn lift_stmt(&mut self, stmt: &Stmt<'db>) -> Vec<Stmt<'db>> {
        match &stmt.kind {
            StmtKind::Let { name, ty } => vec![Stmt {
                span: stmt.span,
                kind: StmtKind::Let {
                    name: name.clone(),
                    ty: ty.clone(),
                },
            }],
            StmtKind::Assign { lhs, rhs } => {
                if let ExprKind::Var(ptr_name) = &lhs.kind {
                    if let ExprKind::Call { callee, args } = &rhs.kind
                        && self.ctx.alloc_mem_fns.contains(callee.as_str())
                        && args.len() == 1
                        && let Some((words_expr, sz_words_opt)) =
                            self.bytes_expr_to_words(args[0].clone())
                    {
                        if let Some(sz_words) = sz_words_opt {
                            self.alloc_vars.insert(ptr_name.as_str().to_owned(), sz_words);
                        }
                        self.used_mem_alloc = true;
                        return vec![Stmt {
                            span: stmt.span,
                            kind: StmtKind::Assign {
                                lhs: lhs.clone(),
                                rhs: Expr {
                                    span: rhs.span,
                                    ty: Ty::word(rhs.span),
                                    kind: ExprKind::Alloc {
                                        words: Box::new(words_expr),
                                    },
                                },
                            },
                        }];
                    }
                    if let ExprKind::Var(src_name) = &rhs.kind
                        && let Some(&sz_words) = self.alloc_vars.get(src_name.as_str())
                    {
                        self.alloc_vars
                            .insert(ptr_name.as_str().to_owned(), sz_words);
                    }
                }
                let lhs = self.lift_expr(lhs.clone());
                let rhs = self.lift_expr(rhs.clone());
                vec![Stmt {
                    span: stmt.span,
                    kind: StmtKind::Assign { lhs, rhs },
                }]
            }
            StmtKind::Expr(expr) => {
                if let Some(store_stmt) = self.try_lift_store_expr(stmt.span, expr) {
                    return vec![store_stmt];
                }
                vec![Stmt {
                    span: stmt.span,
                    kind: StmtKind::Expr(self.lift_expr(expr.clone())),
                }]
            }
            StmtKind::Return(expr) => vec![Stmt {
                span: stmt.span,
                kind: StmtKind::Return(self.lift_expr(expr.clone())),
            }],
            StmtKind::Block(stmts) => vec![Stmt {
                span: stmt.span,
                kind: StmtKind::Block(self.lift_stmts(stmts)),
            }],
            StmtKind::For {
                init,
                cond,
                post,
                body,
            } => vec![Stmt {
                span: stmt.span,
                kind: StmtKind::For {
                    init: self.lift_stmts(init),
                    cond: self.lift_expr(cond.clone()),
                    post: self.lift_stmts(post),
                    body: self.lift_stmts(body),
                },
            }],
            StmtKind::Match {
                target,
                scrutinee,
                alts,
            } => vec![Stmt {
                span: stmt.span,
                kind: StmtKind::Match {
                    target: target.clone(),
                    scrutinee: self.lift_expr(scrutinee.clone()),
                    alts: alts
                        .iter()
                        .map(|alt| Alt {
                            span: alt.span,
                            pat: alt.pat.clone(),
                            binder: alt.binder.clone(),
                            body: self.lift_stmts(&alt.body),
                        })
                        .collect(),
                },
            }],
            StmtKind::Assembly(yul_stmts) => {
                if let Some(lifted_asm) = self.try_lift_assembly(stmt.span, yul_stmts) {
                    return lifted_asm;
                }
                vec![stmt.clone()]
            }
            StmtKind::Break
            | StmtKind::Continue
            | StmtKind::Revert(_)
            | StmtKind::Comment(_)
            | StmtKind::InitHeap
            | StmtKind::ParamMem { .. }
            | StmtKind::StorePlace { .. }
            | StmtKind::SplitPlace { .. }
            | StmtKind::JoinPlace { .. }
            | StmtKind::FreePlace { .. } => vec![stmt.clone()],
        }
    }

    fn try_lift_assembly(
        &mut self,
        span: Span<'db>,
        yul_stmts: &[YulStmt<'db>],
    ) -> Option<Vec<Stmt<'db>>> {
        let [ystmt] = yul_stmts else {
            return None;
        };
        let db = self.db;
        match &ystmt.kind {
            YulStmtKind::Expr(YulExpr {
                kind: YulExprKind::Call { name, args },
                ..
            }) if (*name.atom()).text(db) == "sstore" && args.len() == 2 => {
                let slot = yul_expr_to_hull_expr(db, span, &args[0])?;
                let val = yul_expr_to_hull_expr(db, span, &args[1])?;
                let place = self.storage_place_from_slot(slot);
                Some(vec![Stmt {
                    span,
                    kind: StmtKind::StorePlace {
                        place,
                        value: val,
                        evidence: HullEvidence::Checked,
                    },
                }])
            }
            YulStmtKind::Expr(YulExpr {
                kind: YulExprKind::Call { name, args },
                ..
            }) if (*name.atom()).text(db) == "mstore"
                && args.len() == 2
                && is_yul_lit_u64(&args[0], 0x40)
                && is_yul_init_heap_val(db, &args[1]) =>
            {
                Some(vec![Stmt {
                    span,
                    kind: StmtKind::InitHeap,
                }])
            }
            YulStmtKind::Assign { names, value } if names.len() == 1 => {
                let YulExprKind::Call { name, args } = &value.kind else {
                    return None;
                };
                if (*name.atom()).text(db) == "sload" && args.len() == 1 {
                    let slot = yul_expr_to_hull_expr(db, span, &args[0])?;
                    let place = self.storage_place_from_slot(slot);
                    let word = Ty::word(span);
                    let lhs_name = (*names[0].atom()).text(db).to_owned();
                    return Some(vec![Stmt {
                        span,
                        kind: StmtKind::Assign {
                            lhs: Expr::var(span, lhs_name, word.clone()),
                            rhs: Expr {
                                span,
                                ty: word,
                                kind: ExprKind::LoadPlace {
                                    place,
                                    evidence: HullEvidence::Checked,
                                },
                            },
                        },
                    }]);
                }
                None
            }
            _ => None,
        }
    }

    fn try_lift_store_expr(&mut self, span: Span<'db>, expr: &Expr<'db>) -> Option<Stmt<'db>> {
        let ExprKind::Call { callee, args } = &expr.kind else {
            return None;
        };
        if self.ctx.mstore_fns.contains(callee.as_str()) && args.len() == 2 {
            if matches!(&args[0].kind, ExprKind::Word(w) if is_zero_or_64(w, 0x40))
                && matches!(&args[1].kind, ExprKind::Word(w) if is_zero_or_64(w, 0x80))
            {
                return Some(Stmt {
                    span,
                    kind: StmtKind::InitHeap,
                });
            }
            let (place, evidence) = self.try_mem_place(&args[0])?;
            let value = self.lift_expr(args[1].clone());
            return Some(Stmt {
                span,
                kind: StmtKind::StorePlace {
                    place,
                    value,
                    evidence,
                },
            });
        }
        if self.ctx.sstore_fns.contains(callee.as_str()) && args.len() == 2 {
            let place = self.storage_place_from_slot(args[0].clone());
            let value = self.lift_expr(args[1].clone());
            return Some(Stmt {
                span,
                kind: StmtKind::StorePlace {
                    place,
                    value,
                    evidence: HullEvidence::Checked,
                },
            });
        }
        if callee.as_str() == "tstore" && args.len() == 2 {
            let slot = self.lift_expr(args[0].clone());
            let value = self.lift_expr(args[1].clone());
            return Some(Stmt {
                span,
                kind: StmtKind::StorePlace {
                    place: PlaceExpr::tstor(slot, 1),
                    value,
                    evidence: HullEvidence::Checked,
                },
            });
        }
        None
    }

    fn try_mem_place(&mut self, addr_expr: &Expr<'db>) -> Option<(PlaceExpr<'db>, HullEvidence)> {
        let addr_expr = unwrap_identity_expr(addr_expr, &self.ctx.identity_fns);
        match &addr_expr.kind {
            ExprKind::Var(name) => {
                let &root_words = self.alloc_vars.get(name.as_str())?;
                let place = PlaceExpr::mem(addr_expr.clone(), root_words).field(0, 1);
                Some((place, HullEvidence::Checked))
            }
            ExprKind::Call { callee, args }
                if self.ctx.add_fns.contains(callee.as_str()) && args.len() == 2 =>
            {
                let a0 = unwrap_identity_expr(&args[0], &self.ctx.identity_fns);
                let a1 = unwrap_identity_expr(&args[1], &self.ctx.identity_fns);
                let (base_expr, off_expr) = match (&a0.kind, &a1.kind) {
                    (ExprKind::Var(n), _) if self.alloc_vars.contains_key(n.as_str()) => (a0, a1),
                    (_, ExprKind::Var(n)) if self.alloc_vars.contains_key(n.as_str()) => (a1, a0),
                    _ => return None,
                };
                let ExprKind::Var(base_name) = &base_expr.kind else {
                    unreachable!();
                };
                let root_words = *self.alloc_vars.get(base_name.as_str())?;
                if let ExprKind::Word(off_text) = &off_expr.kind
                    && let Some(byte_off) = parse_u64_lit(off_text)
                    && byte_off % 32 == 0
                {
                    let word_idx = (byte_off / 32) as usize;
                    if word_idx < root_words {
                        let place =
                            PlaceExpr::mem(base_expr.clone(), root_words).field(word_idx, 1);
                        return Some((place, HullEvidence::Checked));
                    }
                }
                let off_lifted = self.lift_expr(off_expr.clone());
                let place = PlaceExpr::mem(base_expr.clone(), root_words).with_proj(
                    PlaceProj::ByteOffset {
                        byte_offset: Box::new(off_lifted),
                        size_words: 1,
                    },
                );
                Some((place, HullEvidence::Trusted))
            }
            _ => None,
        }
    }

    fn storage_place_from_slot(&mut self, slot_expr: Expr<'db>) -> PlaceExpr<'db> {
        let unwrapped = unwrap_identity_expr(&slot_expr, &self.ctx.identity_fns).clone();
        if let ExprKind::Var(name) = &unwrapped.kind
            && let Some((w0, w1)) = self.hash2_vars.get(name.as_str()).cloned()
        {
            return PlaceExpr::stor(w1, 1).entry(w0, 1);
        }
        if let Some((w0, w1)) = self.match_hash2_call(&unwrapped) {
            let w0 = self.lift_expr(w0);
            let w1 = self.lift_expr(w1);
            return PlaceExpr::stor(w1, 1).entry(w0, 1);
        }
        let slot = self.lift_expr(unwrapped);
        PlaceExpr::stor(slot, 1)
    }

    fn lift_expr(&mut self, expr: Expr<'db>) -> Expr<'db> {
        let span = expr.span;
        let ty = expr.ty;
        let kind = match expr.kind {
            ExprKind::Call { callee, args } => {
                if self.ctx.mload_fns.contains(callee.as_str())
                    && args.len() == 1
                    && let Some((place, evidence)) = self.try_mem_place(&args[0])
                {
                    return Expr {
                        span,
                        ty,
                        kind: ExprKind::LoadPlace { place, evidence },
                    };
                }
                if self.ctx.sload_fns.contains(callee.as_str()) && args.len() == 1 {
                    let place = self.storage_place_from_slot(args[0].clone());
                    return Expr {
                        span,
                        ty,
                        kind: ExprKind::LoadPlace {
                            place,
                            evidence: HullEvidence::Checked,
                        },
                    };
                }
                if callee.as_str() == "tload" && args.len() == 1 {
                    let slot = self.lift_expr(args[0].clone());
                    return Expr {
                        span,
                        ty,
                        kind: ExprKind::LoadPlace {
                            place: PlaceExpr::tstor(slot, 1),
                            evidence: HullEvidence::Checked,
                        },
                    };
                }
                if self.ctx.alloc_mem_fns.contains(callee.as_str())
                    && args.len() == 1
                    && let Some((words_expr, _)) = self.bytes_expr_to_words(args[0].clone())
                {
                    self.used_mem_alloc = true;
                    return Expr {
                        span,
                        ty,
                        kind: ExprKind::Alloc {
                            words: Box::new(words_expr),
                        },
                    };
                }
                ExprKind::Call {
                    callee,
                    args: args.into_iter().map(|a| self.lift_expr(a)).collect(),
                }
            }
            ExprKind::Pair(a, b) => {
                ExprKind::Pair(Box::new(self.lift_expr(*a)), Box::new(self.lift_expr(*b)))
            }
            ExprKind::Fst(a) => ExprKind::Fst(Box::new(self.lift_expr(*a))),
            ExprKind::Snd(a) => ExprKind::Snd(Box::new(self.lift_expr(*a))),
            ExprKind::Inl { target, value } => ExprKind::Inl {
                target,
                value: Box::new(self.lift_expr(*value)),
            },
            ExprKind::Inr { target, value } => ExprKind::Inr {
                target,
                value: Box::new(self.lift_expr(*value)),
            },
            ExprKind::InK {
                index,
                target,
                value,
            } => ExprKind::InK {
                index,
                target,
                value: Box::new(self.lift_expr(*value)),
            },
            ExprKind::If {
                target,
                cond,
                then_expr,
                else_expr,
            } => ExprKind::If {
                target,
                cond: Box::new(self.lift_expr(*cond)),
                then_expr: Box::new(self.lift_expr(*then_expr)),
                else_expr: Box::new(self.lift_expr(*else_expr)),
            },
            ExprKind::Alloc { words } => ExprKind::Alloc {
                words: Box::new(self.lift_expr(*words)),
            },
            ExprKind::LoadPlace { place, evidence } => ExprKind::LoadPlace {
                place: self.lift_place(place),
                evidence,
            },
            other @ (ExprKind::Word(_)
            | ExprKind::Bool(_)
            | ExprKind::Unit
            | ExprKind::Var(_)) => other,
        };
        Expr { span, ty, kind }
    }

    fn lift_place(&mut self, place: PlaceExpr<'db>) -> PlaceExpr<'db> {
        PlaceExpr {
            space: place.space,
            base: Box::new(self.lift_expr(*place.base)),
            root_size_words: place.root_size_words,
            projections: place
                .projections
                .into_iter()
                .map(|proj| match proj {
                    PlaceProj::Field {
                        word_offset,
                        field_size_words,
                    } => PlaceProj::Field {
                        word_offset,
                        field_size_words,
                    },
                    PlaceProj::Lane {
                        shift_bits,
                        width_bits,
                    } => PlaceProj::Lane {
                        shift_bits,
                        width_bits,
                    },
                    PlaceProj::Index {
                        index,
                        elem_size_words,
                        bound_elems,
                    } => PlaceProj::Index {
                        index: Box::new(self.lift_expr(*index)),
                        elem_size_words,
                        bound_elems: bound_elems.map(|b| Box::new(self.lift_expr(*b))),
                    },
                    PlaceProj::Entry {
                        key,
                        value_size_words,
                    } => PlaceProj::Entry {
                        key: Box::new(self.lift_expr(*key)),
                        value_size_words,
                    },
                    PlaceProj::ByteOffset {
                        byte_offset,
                        size_words,
                    } => PlaceProj::ByteOffset {
                        byte_offset: Box::new(self.lift_expr(*byte_offset)),
                        size_words,
                    },
                })
                .collect(),
        }
    }
}

fn is_yul_lit_u64(expr: &YulExpr<'_>, expected: u64) -> bool {
    matches!(
        &expr.kind,
        YulExprKind::Lit(YulLitKind::Number(n) | YulLitKind::Hex(n))
            if parse_u64_lit(n) == Some(expected)
    )
}

fn is_yul_init_heap_val(db: &dyn HirDb, expr: &YulExpr<'_>) -> bool {
    if is_yul_lit_u64(expr, 0x80) {
        return true;
    }
    if let YulExprKind::Call { name, args } = &expr.kind
        && (*name.atom()).text(db) == "memoryguard"
        && args.len() == 1
        && is_yul_lit_u64(&args[0], 0x80)
    {
        return true;
    }
    false
}

fn code_stmts_have_init_heap_yul(db: &dyn HirDb, stmts: &[Stmt<'_>]) -> bool {
    stmts.iter().any(|s| {
        if let StmtKind::Assembly(ys) = &s.kind {
            ys.iter().any(|y| {
                matches!(
                    &y.kind,
                    YulStmtKind::Expr(YulExpr {
                        kind: YulExprKind::Call { name, args },
                        ..
                    }) if (*name.atom()).text(db) == "mstore"
                        && args.len() == 2
                        && is_yul_lit_u64(&args[0], 0x40)
                        && is_yul_init_heap_val(db, &args[1])
                )
            })
        } else {
            false
        }
    })
}

fn replace_init_heap_yul<'db>(db: &dyn HirDb, stmts: Vec<Stmt<'db>>) -> Vec<Stmt<'db>> {
    stmts
        .into_iter()
        .map(|s| {
            if let StmtKind::Assembly(ys) = &s.kind
                && ys.len() == 1
                && let YulStmtKind::Expr(YulExpr {
                    kind: YulExprKind::Call { name, args },
                    ..
                }) = &ys[0].kind
                && (*name.atom()).text(db) == "mstore"
                && args.len() == 2
                && is_yul_lit_u64(&args[0], 0x40)
                && is_yul_init_heap_val(db, &args[1])
            {
                Stmt {
                    span: s.span,
                    kind: StmtKind::InitHeap,
                }
            } else {
                s
            }
        })
        .collect()
}

fn yul_expr_to_hull_expr<'db>(
    db: &dyn HirDb,
    span: Span<'db>,
    yexpr: &YulExpr<'db>,
) -> Option<Expr<'db>> {
    match &yexpr.kind {
        YulExprKind::Lit(YulLitKind::Number(n) | YulLitKind::Hex(n)) => {
            Some(Expr::word(span, n.clone()))
        }
        YulExprKind::Ident(ident) => Some(Expr::var(
            span,
            (*ident.atom()).text(db).to_owned(),
            Ty::word(span),
        )),
        YulExprKind::Call { name, args } => {
            let mut hull_args = Vec::with_capacity(args.len());
            for a in args {
                hull_args.push(yul_expr_to_hull_expr(db, span, a)?);
            }
            Some(Expr {
                span,
                ty: Ty::word(span),
                kind: ExprKind::Call {
                    callee: (*name.atom()).text(db).to_owned().into(),
                    args: hull_args,
                },
            })
        }
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Erasure pass: Hull+ -> Core Hull
// ---------------------------------------------------------------------------

#[derive(Default)]
struct Eraser {
    fresh: usize,
    needs_hash2: bool,
}

fn erase_object<'db>(obj: &Object<'db>, eraser: &mut Eraser) -> Object<'db> {
    let mut sub_eraser = Eraser {
        fresh: eraser.fresh,
        needs_hash2: false,
    };
    let stmts = sub_eraser.erase_stmts(&obj.code.stmts);
    let functions = erase_function_list(&obj.code.functions, obj.code.span, &mut sub_eraser);
    eraser.fresh = sub_eraser.fresh;
    let inners = obj
        .inners
        .iter()
        .map(|inner| erase_object(inner, eraser))
        .collect();
    Object {
        span: obj.span,
        name: obj.name.clone(),
        code: CodeBlock {
            span: obj.code.span,
            stmts,
            functions,
        },
        inners,
    }
}

fn erase_function_list<'db>(
    functions: &[Function<'db>],
    scope_span: Span<'db>,
    eraser: &mut Eraser,
) -> Vec<Function<'db>> {
    let mut out: Vec<Function<'db>> = functions
        .iter()
        .map(|f| Function {
            span: f.span,
            name: f.name.clone(),
            args: f.args.clone(),
            ret: f.ret.clone(),
            body: eraser.erase_stmts(&f.body),
        })
        .collect();

    if eraser.needs_hash2
        && !out
            .iter()
            .any(|f| f.name.as_str() == STORAGE_HASH2_HELPER)
    {
        out.push(synthesize_storage_hash2_fn(scope_span));
    }
    eraser.needs_hash2 = false;
    out
}

fn synthesize_storage_hash2_fn<'db>(span: Span<'db>) -> Function<'db> {
    let word = Ty::word(span);
    Function {
        span,
        name: STORAGE_HASH2_HELPER.into(),
        args: vec![
            Arg {
                span,
                name: "x".into(),
                ty: word.clone(),
            },
            Arg {
                span,
                name: "y".into(),
                ty: word.clone(),
            },
        ],
        ret: word.clone(),
        body: vec![
            Stmt {
                span,
                kind: StmtKind::Expr(call_expr(
                    span,
                    Ty::unit(span),
                    "mstore",
                    vec![
                        Expr::word(span, "0"),
                        Expr::var(span, "x", word.clone()),
                    ],
                )),
            },
            Stmt {
                span,
                kind: StmtKind::Expr(call_expr(
                    span,
                    Ty::unit(span),
                    "mstore",
                    vec![
                        Expr::word(span, "32"),
                        Expr::var(span, "y", word.clone()),
                    ],
                )),
            },
            Stmt {
                span,
                kind: StmtKind::Return(call_expr(
                    span,
                    word,
                    "keccak256",
                    vec![Expr::word(span, "0"), Expr::word(span, "64")],
                )),
            },
        ],
    }
}

impl Eraser {
    fn fresh_name(&mut self, prefix: &str) -> String {
        let name = format!("__hull_plus_{prefix}_{}", self.fresh);
        self.fresh += 1;
        name
    }

    fn erase_stmts<'db>(&mut self, stmts: &[Stmt<'db>]) -> Vec<Stmt<'db>> {
        let mut out = Vec::with_capacity(stmts.len());
        for stmt in stmts {
            out.extend(self.erase_stmt(stmt));
        }
        out
    }

    fn erase_stmt<'db>(&mut self, stmt: &Stmt<'db>) -> Vec<Stmt<'db>> {
        let span = stmt.span;
        match &stmt.kind {
            StmtKind::InitHeap => vec![Stmt {
                span,
                kind: StmtKind::Expr(call_expr(
                    span,
                    Ty::unit(span),
                    "mstore",
                    vec![Expr::word(span, "64"), Expr::word(span, "128")],
                )),
            }],
            StmtKind::ParamMem { .. }
            | StmtKind::JoinPlace { .. }
            | StmtKind::FreePlace { .. } => Vec::new(),
            StmtKind::SplitPlace {
                binder,
                place,
                mode: _,
            } => {
                let mut pre = Vec::new();
                let resolved = self.erase_place_addr(span, place, &mut pre);
                pre.push(Stmt {
                    span,
                    kind: StmtKind::Assign {
                        lhs: Expr::var(span, binder.clone(), Ty::word(span)),
                        rhs: resolved.addr,
                    },
                });
                pre
            }
            StmtKind::StorePlace {
                place,
                value,
                evidence: _,
            } => {
                let mut pre = Vec::new();
                let resolved = self.erase_place_addr(span, place, &mut pre);
                let val_erased = self.erase_expr(value, &mut pre);
                let (load_op, store_op) = match place.space {
                    PlaceSpace::Mem | PlaceSpace::Pool => ("mload", "mstore"),
                    PlaceSpace::Stor => ("sload", "sstore"),
                    PlaceSpace::TStor => ("tload", "tstore"),
                };
                let word_to_store = if let Some((shift_bits, width_bits)) = resolved.lane {
                    let old_word = call_expr(
                        span,
                        Ty::word(span),
                        load_op,
                        vec![resolved.addr.clone()],
                    );
                    insert_lane_expr(span, old_word, val_erased, shift_bits, width_bits)
                } else {
                    val_erased
                };
                pre.push(Stmt {
                    span,
                    kind: StmtKind::Expr(call_expr(
                        span,
                        Ty::unit(span),
                        store_op,
                        vec![resolved.addr, word_to_store],
                    )),
                });
                pre
            }
            StmtKind::Assign {
                lhs,
                rhs:
                    Expr {
                        span: alloc_span,
                        kind: ExprKind::Alloc { words },
                        ..
                    },
            } if matches!(lhs.kind, ExprKind::Var(_)) => {
                let mut pre = Vec::new();
                let lhs_erased = self.erase_expr(lhs, &mut pre);
                let bytes_expr = self.words_to_bytes_expr(*alloc_span, words, &mut pre);
                pre.push(Stmt {
                    span,
                    kind: StmtKind::Assign {
                        lhs: lhs_erased.clone(),
                        rhs: call_expr(
                            *alloc_span,
                            Ty::word(*alloc_span),
                            "mload",
                            vec![Expr::word(*alloc_span, "64")],
                        ),
                    },
                });
                let new_fp = add_expr(*alloc_span, lhs_erased, bytes_expr);
                pre.push(Stmt {
                    span,
                    kind: StmtKind::Expr(call_expr(
                        *alloc_span,
                        Ty::unit(*alloc_span),
                        "mstore",
                        vec![Expr::word(*alloc_span, "64"), new_fp],
                    )),
                });
                pre
            }
            StmtKind::Assign { lhs, rhs } => {
                let mut pre = Vec::new();
                let lhs_erased = self.erase_expr(lhs, &mut pre);
                let rhs_erased = self.erase_expr(rhs, &mut pre);
                pre.push(Stmt {
                    span,
                    kind: StmtKind::Assign {
                        lhs: lhs_erased,
                        rhs: rhs_erased,
                    },
                });
                pre
            }
            StmtKind::Expr(expr) => {
                let mut pre = Vec::new();
                let erased = self.erase_expr(expr, &mut pre);
                pre.push(Stmt {
                    span,
                    kind: StmtKind::Expr(erased),
                });
                pre
            }
            StmtKind::Return(expr) => {
                let mut pre = Vec::new();
                let erased = self.erase_expr(expr, &mut pre);
                pre.push(Stmt {
                    span,
                    kind: StmtKind::Return(erased),
                });
                pre
            }
            StmtKind::Block(stmts) => vec![Stmt {
                span,
                kind: StmtKind::Block(self.erase_stmts(stmts)),
            }],
            StmtKind::For {
                init,
                cond,
                post,
                body,
            } => {
                let mut pre = Vec::new();
                let init_erased = self.erase_stmts(init);
                let cond_erased = self.erase_expr(cond, &mut pre);
                let post_erased = self.erase_stmts(post);
                let body_erased = self.erase_stmts(body);
                pre.push(Stmt {
                    span,
                    kind: StmtKind::For {
                        init: init_erased,
                        cond: cond_erased,
                        post: post_erased,
                        body: body_erased,
                    },
                });
                pre
            }
            StmtKind::Match {
                target,
                scrutinee,
                alts,
            } => {
                let mut pre = Vec::new();
                let scrutinee_erased = self.erase_expr(scrutinee, &mut pre);
                let alts_erased = alts
                    .iter()
                    .map(|alt| Alt {
                        span: alt.span,
                        pat: alt.pat.clone(),
                        binder: alt.binder.clone(),
                        body: self.erase_stmts(&alt.body),
                    })
                    .collect();
                pre.push(Stmt {
                    span,
                    kind: StmtKind::Match {
                        target: target.clone(),
                        scrutinee: scrutinee_erased,
                        alts: alts_erased,
                    },
                });
                pre
            }
            StmtKind::Let { .. }
            | StmtKind::Break
            | StmtKind::Continue
            | StmtKind::Assembly(_)
            | StmtKind::Revert(_)
            | StmtKind::Comment(_) => vec![stmt.clone()],
        }
    }

    fn words_to_bytes_expr<'db>(
        &mut self,
        span: Span<'db>,
        words: &Expr<'db>,
        pre: &mut Vec<Stmt<'db>>,
    ) -> Expr<'db> {
        let erased_words = self.erase_expr(words, pre);
        if let ExprKind::Word(text) = &erased_words.kind
            && let Some(w) = parse_u64_lit(text)
        {
            return Expr::word(span, (w * 32).to_string());
        }
        call_expr(
            span,
            Ty::word(span),
            "mul",
            vec![erased_words, Expr::word(span, "32")],
        )
    }

    fn erase_expr<'db>(&mut self, expr: &Expr<'db>, pre: &mut Vec<Stmt<'db>>) -> Expr<'db> {
        let span = expr.span;
        let ty = expr.ty.clone();
        match &expr.kind {
            ExprKind::Alloc { words } => {
                let bytes_expr = self.words_to_bytes_expr(span, words, pre);
                let tmp = self.fresh_name("alloc");
                let word_ty = Ty::word(span);
                let tmp_var = Expr::var(span, tmp.clone(), word_ty.clone());
                pre.push(Stmt {
                    span,
                    kind: StmtKind::Let {
                        name: tmp.into(),
                        ty: word_ty.clone(),
                    },
                });
                pre.push(Stmt {
                    span,
                    kind: StmtKind::Assign {
                        lhs: tmp_var.clone(),
                        rhs: call_expr(
                            span,
                            word_ty.clone(),
                            "mload",
                            vec![Expr::word(span, "64")],
                        ),
                    },
                });
                let new_fp = add_expr(span, tmp_var.clone(), bytes_expr);
                pre.push(Stmt {
                    span,
                    kind: StmtKind::Expr(call_expr(
                        span,
                        Ty::unit(span),
                        "mstore",
                        vec![Expr::word(span, "64"), new_fp],
                    )),
                });
                tmp_var
            }
            ExprKind::LoadPlace { place, evidence: _ } => {
                let resolved = self.erase_place_addr(span, place, pre);
                let load_op = match place.space {
                    PlaceSpace::Mem | PlaceSpace::Pool => "mload",
                    PlaceSpace::Stor => "sload",
                    PlaceSpace::TStor => "tload",
                };
                let raw = call_expr(span, Ty::word(span), load_op, vec![resolved.addr]);
                let loaded = if let Some((shift_bits, width_bits)) = resolved.lane {
                    extract_lane_expr(span, raw, shift_bits, width_bits)
                } else {
                    raw
                };
                Expr {
                    span,
                    ty,
                    kind: loaded.kind,
                }
            }
            ExprKind::Pair(a, b) => Expr {
                span,
                ty,
                kind: ExprKind::Pair(
                    Box::new(self.erase_expr(a, pre)),
                    Box::new(self.erase_expr(b, pre)),
                ),
            },
            ExprKind::Fst(a) => Expr {
                span,
                ty,
                kind: ExprKind::Fst(Box::new(self.erase_expr(a, pre))),
            },
            ExprKind::Snd(a) => Expr {
                span,
                ty,
                kind: ExprKind::Snd(Box::new(self.erase_expr(a, pre))),
            },
            ExprKind::Inl { target, value } => Expr {
                span,
                ty,
                kind: ExprKind::Inl {
                    target: target.clone(),
                    value: Box::new(self.erase_expr(value, pre)),
                },
            },
            ExprKind::Inr { target, value } => Expr {
                span,
                ty,
                kind: ExprKind::Inr {
                    target: target.clone(),
                    value: Box::new(self.erase_expr(value, pre)),
                },
            },
            ExprKind::InK {
                index,
                target,
                value,
            } => Expr {
                span,
                ty,
                kind: ExprKind::InK {
                    index: *index,
                    target: target.clone(),
                    value: Box::new(self.erase_expr(value, pre)),
                },
            },
            ExprKind::Call { callee, args } => {
                let args_erased = args.iter().map(|a| self.erase_expr(a, pre)).collect();
                Expr {
                    span,
                    ty,
                    kind: ExprKind::Call {
                        callee: callee.clone(),
                        args: args_erased,
                    },
                }
            }
            ExprKind::If {
                target,
                cond,
                then_expr,
                else_expr,
            } => Expr {
                span,
                ty,
                kind: ExprKind::If {
                    target: target.clone(),
                    cond: Box::new(self.erase_expr(cond, pre)),
                    then_expr: Box::new(self.erase_expr(then_expr, pre)),
                    else_expr: Box::new(self.erase_expr(else_expr, pre)),
                },
            },
            ExprKind::Word(_) | ExprKind::Bool(_) | ExprKind::Unit | ExprKind::Var(_) => {
                expr.clone()
            }
        }
    }

    fn erase_place_addr<'db>(
        &mut self,
        span: Span<'db>,
        place: &PlaceExpr<'db>,
        pre: &mut Vec<Stmt<'db>>,
    ) -> ErasedPlace<'db> {
        let unit = match place.space {
            PlaceSpace::Mem | PlaceSpace::Pool => 32usize,
            PlaceSpace::Stor | PlaceSpace::TStor => 1usize,
        };
        let mut addr = self.erase_expr(&place.base, pre);
        let mut lane = None;

        for proj in &place.projections {
            match proj {
                PlaceProj::Field {
                    word_offset,
                    field_size_words: _,
                } => {
                    let off = word_offset * unit;
                    if off != 0 {
                        addr = add_expr(span, addr, Expr::word(span, off.to_string()));
                    }
                    lane = None;
                }
                PlaceProj::Lane {
                    shift_bits,
                    width_bits,
                } => {
                    lane = Some((*shift_bits, *width_bits));
                }
                PlaceProj::Index {
                    index,
                    elem_size_words,
                    bound_elems,
                } => {
                    let idx_erased = self.erase_expr(index, pre);
                    if let Some(bound) = bound_elems {
                        let bound_erased = self.erase_expr(bound, pre);
                        let in_bounds = call_expr(
                            span,
                            Ty::word(span),
                            "lt",
                            vec![idx_erased.clone(), bound_erased],
                        );
                        let oob = call_expr(span, Ty::word(span), "iszero", vec![in_bounds]);
                        let bool_sum = Ty::sum(span, Ty::unit(span), Ty::unit(span));
                        pre.push(Stmt {
                            span,
                            kind: StmtKind::Match {
                                target: bool_sum,
                                scrutinee: oob,
                                alts: vec![
                                    Alt {
                                        span,
                                        pat: crate::ir::Pat {
                                            span,
                                            kind: crate::ir::PatKind::Con(crate::ir::Con::Inr),
                                        },
                                        binder: "_oob".into(),
                                        body: vec![Stmt {
                                            span,
                                            kind: StmtKind::Expr(call_expr(
                                                span,
                                                Ty::unit(span),
                                                "revert",
                                                vec![
                                                    Expr::word(span, "0"),
                                                    Expr::word(span, "0"),
                                                ],
                                            )),
                                        }],
                                    },
                                    Alt {
                                        span,
                                        pat: crate::ir::Pat {
                                            span,
                                            kind: crate::ir::PatKind::Con(crate::ir::Con::Inl),
                                        },
                                        binder: "_ok".into(),
                                        body: Vec::new(),
                                    },
                                ],
                            },
                        });
                    }
                    let stride = elem_size_words * unit;
                    let scaled = if stride == 1 {
                        idx_erased
                    } else {
                        call_expr(
                            span,
                            Ty::word(span),
                            "mul",
                            vec![idx_erased, Expr::word(span, stride.to_string())],
                        )
                    };
                    addr = add_expr(span, addr, scaled);
                    lane = None;
                }
                PlaceProj::Entry {
                    key,
                    value_size_words: _,
                } => {
                    let key_erased = self.erase_expr(key, pre);
                    self.needs_hash2 = true;
                    addr = call_expr(
                        span,
                        Ty::word(span),
                        STORAGE_HASH2_HELPER,
                        vec![key_erased, addr],
                    );
                    lane = None;
                }
                PlaceProj::ByteOffset {
                    byte_offset,
                    size_words: _,
                } => {
                    let off_erased = self.erase_expr(byte_offset, pre);
                    addr = add_expr(span, addr, off_erased);
                    lane = None;
                }
            }
        }

        ErasedPlace { addr, lane }
    }
}

struct ErasedPlace<'db> {
    addr: Expr<'db>,
    lane: Option<(usize, usize)>,
}

fn call_expr<'db>(
    span: Span<'db>,
    ty: Ty<'db>,
    callee: impl Into<Name>,
    args: Vec<Expr<'db>>,
) -> Expr<'db> {
    Expr {
        span,
        ty,
        kind: ExprKind::Call {
            callee: callee.into(),
            args,
        },
    }
}

fn add_expr<'db>(span: Span<'db>, lhs: Expr<'db>, rhs: Expr<'db>) -> Expr<'db> {
    if matches!(&rhs.kind, ExprKind::Word(w) if parse_u64_lit(w) == Some(0)) {
        return lhs;
    }
    if matches!(&lhs.kind, ExprKind::Word(w) if parse_u64_lit(w) == Some(0)) {
        return rhs;
    }
    call_expr(span, Ty::word(span), "add", vec![lhs, rhs])
}

fn lane_mask_hex(width_bits: usize) -> String {
    if width_bits >= 256 {
        "0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff".to_owned()
    } else {
        let bytes = width_bits / 8;
        format!("0x{}", "ff".repeat(bytes.max(1)))
    }
}

fn extract_lane_expr<'db>(
    span: Span<'db>,
    word_expr: Expr<'db>,
    shift_bits: usize,
    width_bits: usize,
) -> Expr<'db> {
    let shifted = if shift_bits == 0 {
        word_expr
    } else {
        call_expr(
            span,
            Ty::word(span),
            "shr",
            vec![Expr::word(span, shift_bits.to_string()), word_expr],
        )
    };
    if width_bits >= 256 {
        shifted
    } else {
        call_expr(
            span,
            Ty::word(span),
            "and",
            vec![shifted, Expr::word(span, lane_mask_hex(width_bits))],
        )
    }
}

fn insert_lane_expr<'db>(
    span: Span<'db>,
    old_word: Expr<'db>,
    new_val: Expr<'db>,
    shift_bits: usize,
    width_bits: usize,
) -> Expr<'db> {
    if shift_bits == 0 && width_bits >= 256 {
        return new_val;
    }
    let mask = Expr::word(span, lane_mask_hex(width_bits));
    let masked_val = call_expr(span, Ty::word(span), "and", vec![new_val, mask.clone()]);
    let shifted_val = if shift_bits == 0 {
        masked_val
    } else {
        call_expr(
            span,
            Ty::word(span),
            "shl",
            vec![Expr::word(span, shift_bits.to_string()), masked_val],
        )
    };
    let shifted_mask = if shift_bits == 0 {
        mask
    } else {
        call_expr(
            span,
            Ty::word(span),
            "shl",
            vec![Expr::word(span, shift_bits.to_string()), mask],
        )
    };
    let inv_mask = call_expr(span, Ty::word(span), "not", vec![shifted_mask]);
    let cleared = call_expr(span, Ty::word(span), "and", vec![old_word, inv_mask]);
    call_expr(span, Ty::word(span), "or", vec![cleared, shifted_val])
}
