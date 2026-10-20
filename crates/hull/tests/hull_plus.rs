use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    fs,
    path::{Path, PathBuf},
};

use hir::{anchor::DefLocationTable, input::SourceFile, span::Span};
use nameres::{
    LibraryId, ModuleFileSnapshot, ModuleFsSnapshot, ModuleId, ModuleKey, ModuleTree,
    module_id_from_key, module_key_for_path, module_path_display, resolve_module_path_candidate,
};
use parser::parse_file_to_hir;
use rustc_hash::{FxHashMap, FxHashSet};
use salsa::Setter;
use solcore_hull::{
    Arg, BorrowMode, CheckDiagnosticKind, CodeBlock, EmitDiagnostic, EmitDiagnosticKind,
    EmitOptions, Expr, ExprKind, Function, HullEvidence, Object, PlaceExpr, Program, Stmt,
    StmtKind, Ty, check_program_with_db, emit_module, emit_module_hull_plus, erase_program,
    has_hull_plus, pretty_program,
};
use specialize::{SpecializeOptions, SpecializeOutput, specialize_module};

#[salsa::db]
#[derive(Default, Clone)]
struct TestDb {
    storage: salsa::Storage<Self>,
    module_tree: Option<ModuleTree>,
    module_fs_snapshot: Option<ModuleFsSnapshot>,
    module_file_snapshot: Option<ModuleFileSnapshot>,
    module_files: FxHashMap<ModuleKey, SourceFile>,
}

impl TestDb {
    fn insert_module_file(&mut self, key: ModuleKey, file: SourceFile) {
        if self.module_files.insert(key, file) == Some(file) {
            return;
        }
        let files = self
            .module_files
            .iter()
            .map(|(key, file)| (key.clone(), *file))
            .collect();
        if let Some(snapshot) = self.module_file_snapshot {
            snapshot.set_files(self).to(files);
        } else {
            self.module_file_snapshot = Some(ModuleFileSnapshot::new(self, files));
        }
    }
}

#[salsa::db]
impl salsa::Database for TestDb {}

#[salsa::db]
impl hir::Db for TestDb {
    fn def_location_table<'db>(&'db self, file: SourceFile) -> &'db DefLocationTable<'db> {
        parse_file_to_hir(self, file).def_locations(self)
    }
}

#[salsa::db]
impl parser::Db for TestDb {}

#[salsa::db]
impl nameres::Db for TestDb {
    fn module_tree(&self) -> ModuleTree {
        self.module_tree.unwrap_or_else(|| {
            ModuleTree::new(
                self,
                PathBuf::from("/main"),
                PathBuf::from("/std"),
                BTreeMap::new(),
            )
        })
    }

    fn module_fs_snapshot(&self) -> ModuleFsSnapshot {
        self.module_fs_snapshot
            .unwrap_or_else(|| ModuleFsSnapshot::new(self, BTreeSet::new(), BTreeMap::new()))
    }

    fn module_file_snapshot(&self) -> ModuleFileSnapshot {
        self.module_file_snapshot
            .unwrap_or_else(|| ModuleFileSnapshot::new(self, BTreeMap::new()))
    }

    fn module_file<'db>(&'db self, module: ModuleId<'db>) -> Option<SourceFile> {
        self.module_file_snapshot()
            .files(self)
            .get(&module.key(self))
            .copied()
    }
}

#[salsa::db]
impl hir_ty::Db for TestDb {}

#[test]
fn storage_counter_and_mapping_fixtures_lift_to_hull_plus_and_erase_cleanly() {
    for (stem, rel, expected_plus_snippets) in [
        (
            "basic_counter",
            "spec/120basicCounter.sol",
            &[
                "__hull_load_place(__hull_place_stor(0, 1), checked)",
                "__hull_store_place(__hull_place_stor(0, 1),",
            ][..],
        ),
        (
            "counters",
            "spec/122counters.sol",
            &[
                "__hull_load_place(__hull_place_stor(0, 1), checked)",
                "__hull_load_place(__hull_place_stor(2, 1), checked)",
                "__hull_store_place(__hull_place_stor(0, 1),",
                "__hull_store_place(__hull_place_stor(2, 1),",
            ][..],
        ),
        (
            "nanoerc20",
            "spec/126nanoerc20.sol",
            &[
                "__hull_load_place(__hull_place_stor(2, 1), checked)",
                "__hull_load_place(__hull_entry(__hull_place_stor(",
                "__hull_store_place(__hull_entry(__hull_place_stor(",
            ][..],
        ),
    ] {
        let (plus_text, erased_text) = compile_fixture_hull_plus_and_erased(stem, rel);
        for snippet in expected_plus_snippets {
            assert!(
                plus_text.contains(snippet),
                "{rel}: missing expected Hull+ snippet {snippet:?} in:\n{plus_text}"
            );
        }
        assert!(
            !erased_text.contains("__hull_"),
            "{rel}: erased Core Hull still contains __hull_ intrinsics:\n{erased_text}"
        );
    }
}

#[test]
fn memory_alloc_and_hash_fixtures_lift_to_hull_plus_and_erase_cleanly() {
    for (stem, rel, expected_plus_snippets) in [
        (
            "string_lit_mem",
            "comptime/string-lit-mem.sol",
            &[
                "__hull_init_heap(128)",
                "__hull_alloc(64, checked)",
                "__hull_store_place(__hull_field(__hull_place_mem(p, 64), 0, 32), 4, checked)",
                "__hull_store_place(__hull_field(__hull_place_mem(p, 64), 32, 32),",
            ][..],
        ),
        (
            "hashes",
            "dispatch/hashes.sol",
            &[
                "__hull_init_heap(128)",
                "__hull_alloc(64, checked)",
                "__hull_store_place(__hull_field(__hull_place_mem(p, 64), 0, 32), 3, checked)",
                "__hull_store_place(__hull_field(__hull_place_mem(p, 64), 32, 32),",
            ][..],
        ),
    ] {
        let (plus_text, erased_text) = compile_fixture_hull_plus_and_erased(stem, rel);
        for snippet in expected_plus_snippets {
            assert!(
                plus_text.contains(snippet),
                "{rel}: missing expected Hull+ snippet {snippet:?} in:\n{plus_text}"
            );
        }
        assert!(
            !erased_text.contains("__hull_"),
            "{rel}: erased Core Hull still contains __hull_ intrinsics:\n{erased_text}"
        );
    }
}

#[test]
fn borrow_split_join_param_mem_and_free_roundtrip() {
    let (db, program) = build_local_struct_sum_hull_plus_program();
    assert!(has_hull_plus(&program));
    assert_eq!(
        check_program_with_db(db, &program),
        Vec::new(),
        "Hull+ local_struct_sum program must check cleanly"
    );
    let plus_text = pretty_program(db, &program);
    for needle in [
        "__hull_param_mem(p, 64)",
        "__hull_init_heap(128)",
        "__hull_alloc(64, checked)",
        "__hull_split_place(__hull_place_mem(p, 64), mutable, checked)",
        "__hull_store_place(__hull_field(__hull_place_mem(b, 64), 0, 32), x, checked)",
        "__hull_store_place(__hull_field(__hull_place_mem(b, 64), 32, 32), y, checked)",
        "__hull_join_place(b)",
        "__hull_free_place(p, 64)",
    ] {
        assert!(
            plus_text.contains(needle),
            "missing {needle:?} in pretty Hull+:\n{plus_text}"
        );
    }

    let erased = erase_program(db, &program);
    assert!(!has_hull_plus(&erased));
    assert_eq!(
        check_program_with_db(db, &erased),
        Vec::new(),
        "Erased Core Hull local_struct_sum program must check cleanly"
    );
    let erased_text = pretty_program(db, &erased);
    assert!(!erased_text.contains("__hull_"), "{erased_text}");
    assert!(erased_text.contains("mstore(64, 128)"), "{erased_text}");
    assert!(erased_text.contains("mload(64)"), "{erased_text}");
}

#[test]
fn corpus_subset_lifts_to_hull_plus_and_erases_without_regressions() {
    std::thread::Builder::new()
        .stack_size(32 * 1024 * 1024)
        .spawn(|| {
            let fixtures = [
                "spec/01id.sol",
                "spec/00answer.sol",
                "spec/022add.sol",
                "spec/024arith.sol",
                "spec/031maybe.sol",
                "spec/041pair.sol",
                "spec/120basicCounter.sol",
                "spec/121counter.sol",
                "spec/122counters.sol",
                "spec/123stackAndStorage.sol",
                "spec/126nanoerc20.sol",
                "spec/127microerc20.sol",
                "spec/128minierc20.sol",
                "comptime/string-lit-mem.sol",
                "dispatch/hashes.sol",
                "dispatch/basic.sol",
                "dispatch/memory.sol",
                "dispatch/storage.sol",
                "dispatch/ownable.sol",
                "dispatch/miniERC20.sol",
                "dispatch/weth9.sol",
            ];
            for (idx, rel) in fixtures.iter().enumerate() {
                let stem = format!("fixture_{idx}");
                let (_plus_text, erased_text) = compile_fixture_hull_plus_and_erased(&stem, rel);
                assert!(
                    !erased_text.contains("__hull_"),
                    "{rel}: unexpected __hull_ intrinsic after erasure"
                );
            }
        })
        .expect("spawn test thread")
        .join()
        .expect("corpus_subset_lifts_to_hull_plus_and_erases_without_regressions");
}

#[test]
fn export_hull_plus_evaluation_benchmarks() {
    let out_dir = Path::new("/lab/argot/scratch/leadsheet-hull-plus/exported");
    if fs::create_dir_all(out_dir).is_err() {
        return;
    }

    let benchmarks = [
        ("basic_counter", "spec/120basicCounter.sol", Some("120basicCounter_")),
        ("counters", "spec/122counters.sol", Some("122counters_")),
        ("nanoerc20", "spec/126nanoerc20.sol", Some("126nanoerc20_")),
        ("string_lit_mem", "comptime/string-lit-mem.sol", None),
        ("hashes", "dispatch/hashes.sol", None),
    ];
    for (stem, rel, digit_prefix) in benchmarks {
        let (mut plus_text, mut erased_text) = compile_fixture_hull_plus_and_erased(stem, rel);
        if let Some(pfx) = digit_prefix {
            let replacement = format!("{stem}_");
            plus_text = plus_text.replace(pfx, &replacement);
            erased_text = erased_text.replace(pfx, &replacement);
        }
        fs::write(out_dir.join(format!("{stem}.plus.hull")), &plus_text)
            .expect("write plus.hull");
        fs::write(out_dir.join(format!("{stem}.erased.hull")), &erased_text)
            .expect("write erased.hull");
    }

    let (db, local_struct_prog) = build_local_struct_sum_hull_plus_program();
    let local_plus = pretty_program(db, &local_struct_prog);
    let local_erased = pretty_program(db, &erase_program(db, &local_struct_prog));
    fs::write(out_dir.join("local_struct_sum.plus.hull"), &local_plus)
        .expect("write local_struct_sum.plus.hull");
    fs::write(out_dir.join("local_struct_sum.erased.hull"), &local_erased)
        .expect("write local_struct_sum.erased.hull");
}

fn build_local_struct_sum_hull_plus_program() -> (&'static TestDb, Program<'static>) {
    let (db, output) = specialize_src_with_std(
        "local_struct_seed",
        "contract Seed { function main() public returns (word) { return 0; } }",
    );
    let span: Span<'static> = emit_module(db, &output.module, EmitOptions::default()).program.span;
    let word = Ty::word(span);

    let read_pair_sum = Function {
        span,
        name: "read_pair_sum".into(),
        args: vec![Arg {
            span,
            name: "p".into(),
            ty: word.clone(),
        }],
        ret: word.clone(),
        body: vec![
            Stmt {
                span,
                kind: StmtKind::ParamMem {
                    param: "p".into(),
                    size_words: 2,
                },
            },
            Stmt {
                span,
                kind: StmtKind::Let {
                    name: "a".into(),
                    ty: word.clone(),
                },
            },
            Stmt {
                span,
                kind: StmtKind::Assign {
                    lhs: Expr::var(span, "a", word.clone()),
                    rhs: Expr {
                        span,
                        ty: word.clone(),
                        kind: ExprKind::LoadPlace {
                            place: PlaceExpr::mem(Expr::var(span, "p", word.clone()), 2)
                                .field(0, 1),
                            evidence: HullEvidence::Checked,
                        },
                    },
                },
            },
            Stmt {
                span,
                kind: StmtKind::Let {
                    name: "b_val".into(),
                    ty: word.clone(),
                },
            },
            Stmt {
                span,
                kind: StmtKind::Assign {
                    lhs: Expr::var(span, "b_val", word.clone()),
                    rhs: Expr {
                        span,
                        ty: word.clone(),
                        kind: ExprKind::LoadPlace {
                            place: PlaceExpr::mem(Expr::var(span, "p", word.clone()), 2)
                                .field(1, 1),
                            evidence: HullEvidence::Checked,
                        },
                    },
                },
            },
            Stmt {
                span,
                kind: StmtKind::Return(Expr {
                    span,
                    ty: word.clone(),
                    kind: ExprKind::Call {
                        callee: "add".into(),
                        args: vec![
                            Expr::var(span, "a", word.clone()),
                            Expr::var(span, "b_val", word.clone()),
                        ],
                    },
                }),
            },
        ],
    };

    let main_fn = Function {
        span,
        name: "main".into(),
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
                kind: StmtKind::InitHeap,
            },
            Stmt {
                span,
                kind: StmtKind::Let {
                    name: "p".into(),
                    ty: word.clone(),
                },
            },
            Stmt {
                span,
                kind: StmtKind::Assign {
                    lhs: Expr::var(span, "p", word.clone()),
                    rhs: Expr {
                        span,
                        ty: word.clone(),
                        kind: ExprKind::Alloc {
                            words: Box::new(Expr::word(span, "2")),
                        },
                    },
                },
            },
            Stmt {
                span,
                kind: StmtKind::Let {
                    name: "b".into(),
                    ty: word.clone(),
                },
            },
            Stmt {
                span,
                kind: StmtKind::SplitPlace {
                    binder: "b".into(),
                    place: PlaceExpr::mem(Expr::var(span, "p", word.clone()), 2),
                    mode: BorrowMode::Mutable,
                },
            },
            Stmt {
                span,
                kind: StmtKind::StorePlace {
                    place: PlaceExpr::mem(Expr::var(span, "b", word.clone()), 2).field(0, 1),
                    value: Expr::var(span, "x", word.clone()),
                    evidence: HullEvidence::Checked,
                },
            },
            Stmt {
                span,
                kind: StmtKind::StorePlace {
                    place: PlaceExpr::mem(Expr::var(span, "b", word.clone()), 2).field(1, 1),
                    value: Expr::var(span, "y", word.clone()),
                    evidence: HullEvidence::Checked,
                },
            },
            Stmt {
                span,
                kind: StmtKind::JoinPlace {
                    place: PlaceExpr::mem(Expr::var(span, "b", word.clone()), 2),
                },
            },
            Stmt {
                span,
                kind: StmtKind::Let {
                    name: "res".into(),
                    ty: word.clone(),
                },
            },
            Stmt {
                span,
                kind: StmtKind::Assign {
                    lhs: Expr::var(span, "res", word.clone()),
                    rhs: Expr {
                        span,
                        ty: word.clone(),
                        kind: ExprKind::Call {
                            callee: "read_pair_sum".into(),
                            args: vec![Expr::var(span, "p", word.clone())],
                        },
                    },
                },
            },
            Stmt {
                span,
                kind: StmtKind::FreePlace {
                    place: PlaceExpr::mem(Expr::var(span, "p", word.clone()), 2),
                },
            },
            Stmt {
                span,
                kind: StmtKind::Return(Expr::var(span, "res", word.clone())),
            },
        ],
    };

    let entry_fn = Function {
        span,
        name: "local_struct_sum_LocalStructSum_main".into(),
        args: Vec::new(),
        ret: word.clone(),
        body: vec![Stmt {
            span,
            kind: StmtKind::Return(Expr {
                span,
                ty: word,
                kind: ExprKind::Call {
                    callee: "main".into(),
                    args: vec![Expr::word(span, "10"), Expr::word(span, "32")],
                },
            }),
        }],
    };

    let program = Program {
        span,
        entry_points: vec!["local_struct_sum_LocalStructSum_main".into(), "main".into()],
        functions: Vec::new(),
        objects: vec![Object {
            span,
            name: "LocalStructSumDeploy".into(),
            code: CodeBlock {
                span,
                stmts: Vec::new(),
                functions: Vec::new(),
            },
            inners: vec![Object {
                span,
                name: "LocalStructSum".into(),
                code: CodeBlock {
                    span,
                    stmts: Vec::new(),
                    functions: vec![read_pair_sum, main_fn, entry_fn],
                },
                inners: Vec::new(),
            }],
        }],
    };
    (db, program)
}

fn compile_fixture_hull_plus_and_erased(_stem: &str, relative: &str) -> (String, String) {
    let fixture = repo_root()
        .join("crates/parser/tests/fixtures/corpus/ok/test/examples")
        .join(relative);
    let (db, output) = specialize_fixture(&fixture, fixture.parent().unwrap());
    assert_eq!(
        output.diagnostics,
        Vec::new(),
        "specialize diagnostics for {relative:?}"
    );
    let options = EmitOptions {
        emit_dispatcher_comments: false,
    };
    let core_emitted = emit_module(db, &output.module, options);
    let plus_emitted = emit_module_hull_plus(db, &output.module, options);

    let non_dispatch: Vec<_> = plus_emitted
        .diagnostics
        .iter()
        .filter(|d| !matches!(d.kind, EmitDiagnosticKind::UnsupportedDispatchEntry { .. }))
        .collect();
    assert_eq!(
        non_dispatch,
        Vec::<&EmitDiagnostic>::new(),
        "emit diagnostics for {relative:?}"
    );

    let core_diags = check_program_with_db(db, &core_emitted.program);
    let plus_diags = check_program_with_db(db, &plus_emitted.program);
    let unexpected_plus: Vec<_> = plus_diags
        .iter()
        .filter(|d| {
            !matches!(
                d.kind,
                CheckDiagnosticKind::ExprAnnotationMismatch { .. }
                    | CheckDiagnosticKind::TypeMismatch { .. }
            )
        })
        .collect();
    assert!(
        unexpected_plus.is_empty() && plus_diags.len() <= core_diags.len(),
        "unexpected Hull+ check diagnostics for {relative:?}: {plus_diags:?}"
    );

    let erased_program = erase_program(db, &plus_emitted.program);
    assert!(
        !has_hull_plus(&erased_program),
        "erased program for {relative:?} still has Hull+ nodes"
    );
    let erased_diags = check_program_with_db(db, &erased_program);
    let unexpected_erased: Vec<_> = erased_diags
        .iter()
        .filter(|d| {
            !matches!(
                d.kind,
                CheckDiagnosticKind::ExprAnnotationMismatch { .. }
                    | CheckDiagnosticKind::TypeMismatch { .. }
            )
        })
        .collect();
    assert!(
        unexpected_erased.is_empty() && erased_diags.len() <= core_diags.len(),
        "unexpected erased check diagnostics for {relative:?}: {erased_diags:?}"
    );

    (
        pretty_program(db, &plus_emitted.program),
        pretty_program(db, &erased_program),
    )
}

fn specialize_src_with_std(name: &str, src: &str) -> (&'static TestDb, SpecializeOutput<'static>) {
    let main_root = PathBuf::from("/lab/argot/scratch/hull-plus-target/hull-plus-tmp").join(name);
    fs::create_dir_all(&main_root).expect("create temp main root");
    let path = main_root.join(format!("{name}.sol"));
    fs::write(&path, src).expect("write temp source");
    specialize_fixture(&path, &main_root)
}

fn specialize_fixture(
    path: &Path,
    main_root: &Path,
) -> (&'static TestDb, SpecializeOutput<'static>) {
    let db = Box::leak(Box::new(TestDb::default()));
    let main_root = main_root.to_path_buf();
    let repo = repo_root();
    let std_root = repo.join("crates/parser/tests/fixtures/corpus/ok/std");
    db.module_tree = Some(ModuleTree::new(
        db,
        main_root.clone(),
        std_root.clone(),
        BTreeMap::new(),
    ));
    db.module_fs_snapshot = Some(module_fs_snapshot_for_roots(
        db,
        [main_root.as_path(), std_root.as_path()],
    ));
    let source = fs::read_to_string(path).expect("fixture source");
    let key =
        module_key_for_path(LibraryId::Main, &main_root, path).expect("fixture under main root");
    let file = SourceFile::new(
        db,
        url::Url::from_file_path(path).expect("file URL"),
        Some(source),
    );
    db.insert_module_file(key.clone(), file);
    let unresolved = load_reachable_modules(db, key);
    assert!(unresolved.is_empty(), "{unresolved:?}");
    let module = parse_file_to_hir(db, file).module(db);
    let output = specialize_module(db, module, SpecializeOptions::default());
    (db, output)
}

fn module_fs_snapshot_for_roots<'a>(
    db: &TestDb,
    roots: impl IntoIterator<Item = &'a Path>,
) -> ModuleFsSnapshot {
    let mut existing_files = BTreeSet::new();
    let mut sibling_stems = BTreeMap::<PathBuf, BTreeSet<String>>::new();
    for root in roots {
        collect_module_fs_snapshot(root, &mut existing_files, &mut sibling_stems);
    }
    let sibling_stems = sibling_stems
        .into_iter()
        .map(|(parent, stems)| (parent, stems.into_iter().collect()))
        .collect();
    ModuleFsSnapshot::new(db, existing_files, sibling_stems)
}

fn collect_module_fs_snapshot(
    dir: &Path,
    existing_files: &mut BTreeSet<PathBuf>,
    sibling_stems: &mut BTreeMap<PathBuf, BTreeSet<String>>,
) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|extension| extension.to_str()) == Some("sol") {
            if path.is_file() {
                existing_files.insert(path.clone());
            }
            if let Some(stem) = path.file_stem().and_then(|stem| stem.to_str()) {
                sibling_stems
                    .entry(dir.to_path_buf())
                    .or_default()
                    .insert(stem.to_owned());
            }
        }
        if path.is_dir() {
            collect_module_fs_snapshot(&path, existing_files, sibling_stems);
        }
    }
}

fn load_reachable_modules(db: &mut TestDb, entry: ModuleKey) -> Vec<String> {
    let mut queue = VecDeque::from([entry]);
    let mut visited = FxHashSet::default();
    let mut unresolved = Vec::new();

    while let Some(key) = queue.pop_front() {
        if !visited.insert(key.clone()) {
            continue;
        }
        let Some(file) = db.module_files.get(&key).copied() else {
            continue;
        };
        let targets = {
            let module = module_id_from_key(&*db, &key);
            let refs = nameres::module_imports(&*db, file);
            refs.import_refs
                .into_iter()
                .chain(refs.export_refs)
                .filter_map(
                    |path| match resolve_module_path_candidate(&*db, module, &path) {
                        Ok(resolved) => Some((resolved.module.key(&*db), resolved.file_path)),
                        Err(_) => {
                            unresolved.push(format!(
                                "{} imports `{}`",
                                module.display(&*db),
                                module_path_display(&*db, &path)
                            ));
                            None
                        }
                    },
                )
                .collect::<Vec<_>>()
        };
        for (target_key, file_path) in targets {
            if !db.module_files.contains_key(&target_key) {
                match fs::read_to_string(&file_path) {
                    Ok(source) => {
                        let file = SourceFile::new(
                            db,
                            url::Url::from_file_path(&file_path).expect("file URL"),
                            Some(source),
                        );
                        db.insert_module_file(target_key.clone(), file);
                    }
                    Err(err) => unresolved.push(format!("{}: {err}", file_path.display())),
                }
            }
            queue.push_back(target_key);
        }
    }
    unresolved
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("crate is under repo/crates/hull")
        .to_path_buf()
}
