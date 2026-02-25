// ╔══════════════════════════════════════════════════════════════════╗
// ║   BATTLE-TESTED CYPHER SYNTAX CONTRACT                         ║
// ║   Auto-generated from corpus analysis + VSA projection         ║
// ╚══════════════════════════════════════════════════════════════════╝

// ═══ SECTION 1: CYPHER VERB CONTRACT ═══
// Each verb classified by Pearl's Causal Ladder:
//   Rung 0 = Meta (schema operations)
//   Rung 1 = Association (observation, read-only)
//   Rung 2 = Intervention (do-calculus, writes)
//   Rung 3 = Counterfactual (deletion, "what if not?")

// [  SPEC] Rung 1 │ MATCH                │ Pattern match (read)
// [  SPEC] Rung 1 │ OPTIONAL MATCH       │ Pattern match with NULL fill
// [  SPEC] Rung 1 │ WHERE                │ Filter predicate
// [  SPEC] Rung 1 │ RETURN               │ Project columns
// [  SPEC] Rung 1 │ WITH                 │ Pipeline intermediate results
// [  SPEC] Rung 1 │ UNWIND               │ Flatten list to rows
// [  SPEC] Rung 1 │ ORDER BY             │ Sort results (ASC|DESC)
// [  SPEC] Rung 1 │ SKIP                 │ Skip N rows
// [  SPEC] Rung 1 │ LIMIT                │ Limit to N rows
// [  SPEC] Rung 1 │ DISTINCT             │ Deduplicate results
// [  SPEC] Rung 2 │ CREATE               │ Create nodes/edges (intervention)
// [TESTED] Rung 2 │ MERGE                │ Upsert: create if not exists (1946 occurrences)
// [  SPEC] Rung 2 │ SET                  │ Set properties/labels
// [  SPEC] Rung 2 │ REMOVE               │ Remove properties/labels
// [  SPEC] Rung 3 │ DELETE               │ Delete nodes/edges
// [  SPEC] Rung 3 │ DETACH DELETE        │ Delete node + all edges
// [  SPEC] Rung 0 │ CREATE INDEX         │ Create B-tree/vector index
// [  SPEC] Rung 0 │ CREATE CONSTRAINT    │ Create uniqueness/existence constraint
// [  SPEC] Rung 0 │ DROP INDEX           │ Drop index
// [  SPEC] Rung 0 │ DROP CONSTRAINT      │ Drop constraint
// [  SPEC] Rung 1 │ CALL                 │ Invoke procedure
// [  SPEC] Rung 1 │ YIELD                │ Bind procedure output columns


// ═══ SECTION 2: RELATIONSHIP TYPE CONTRACT ═══
// Each relationship type with causal rung, direction, and BF16 predicate.

// [  SPEC] Rung 2 │ CAUSES               │ dir=forward    │ 0 edges
// [  SPEC] Rung 2 │ IS_CAUSED_BY         │ dir=backward   │ 0 edges
// [  SPEC] Rung 2 │ TRANSFORMS           │ dir=forward    │ 0 edges
// [  SPEC] Rung 3 │ DISSOLVES_INTO       │ dir=forward    │ 0 edges
// [TESTED] Rung 1 │ NIB4_NEAR            │ dir=symmetric  │ 524 edges
// [TESTED] Rung 1 │ BERT_NEAR            │ dir=symmetric  │ 548 edges
// [TESTED] Rung 1 │ STRUCTURAL_TRUTH     │ dir=symmetric  │ 77 edges
// [TESTED] Rung 2 │ CADENCE_TRUTH        │ dir=forward    │ 50 edges
// [TESTED] Rung 1 │ SURFACE_SYNONYMY     │ dir=symmetric  │ 50 edges
// [TESTED] Rung 1 │ BELONGS_TO           │ dir=forward    │ 219 edges
// [TESTED] Rung 1 │ HAS_MODE             │ dir=forward    │ 219 edges
// [  SPEC] Rung 1 │ VALID_FOR            │ dir=forward    │ 0 edges
// [  SPEC] Rung 2 │ DEVELOPED_BY         │ dir=backward   │ 0 edges
// [  SPEC] Rung 2 │ DEPLOYED_BY          │ dir=backward   │ 0 edges
// [  SPEC] Rung 2 │ OPERATED_BY          │ dir=backward   │ 0 edges
// [  SPEC] Rung 2 │ SUPPLIED_BY          │ dir=backward   │ 0 edges
// [  SPEC] Rung 2 │ INVESTED_IN          │ dir=forward    │ 0 edges
// [  SPEC] Rung 3 │ TARGETS              │ dir=forward    │ 0 edges


// ═══ SECTION 3: NODE LABEL CONTRACT ═══
// [TESTED] QualiaItem                     │ 219 nodes
// [TESTED] Family                         │ 38 nodes
// [TESTED] Mode                           │ 2 nodes


// ═══ SECTION 4: VSA PREDICATE DISTANCE MATRIX ═══
// BF16 structured distance between relationship type predicate vectors.
// sign_flips = causal direction change, exp_shifts = magnitude change

// d=    3 │ NIB4_NEAR            ↔ BERT_NEAR            │ sign=0 exp=0 │ NEAR-SYNONYM (same rung 1, same direction, d=3)
// d=   22 │ BELONGS_TO           ↔ VALID_FOR            │ sign=0 exp=0 │ NEAR-SYNONYM (same rung 1, same direction, d=22)
// d=   49 │ DEVELOPED_BY         ↔ OPERATED_BY          │ sign=0 exp=2 │ NEAR-SYNONYM (same rung 2, same direction, d=49)
// d=   57 │ OPERATED_BY          ↔ INVESTED_IN          │ sign=0 exp=2 │ RELATED (d=57, 0 sign flips)
// d=   80 │ DEVELOPED_BY         ↔ SUPPLIED_BY          │ sign=0 exp=4 │ NEAR-SYNONYM (same rung 2, same direction, d=80)
// d=   88 │ DEVELOPED_BY         ↔ INVESTED_IN          │ sign=0 exp=4 │ RELATED (d=88, 0 sign flips)
// d=   91 │ STRUCTURAL_TRUTH     ↔ HAS_MODE             │ sign=0 exp=4 │ RELATED (d=91, 0 sign flips)
// d=   99 │ BERT_NEAR            ↔ SURFACE_SYNONYMY     │ sign=0 exp=5 │ NEAR-SYNONYM (same rung 1, same direction, d=99)
// d=  100 │ NIB4_NEAR            ↔ SURFACE_SYNONYMY     │ sign=0 exp=5 │ NEAR-SYNONYM (same rung 1, same direction, d=100)
// d=  112 │ BERT_NEAR            ↔ HAS_MODE             │ sign=0 exp=6 │ RELATED (d=112, 0 sign flips)
// d=  113 │ NIB4_NEAR            ↔ HAS_MODE             │ sign=0 exp=6 │ RELATED (d=113, 0 sign flips)
// d=  115 │ OPERATED_BY          ↔ SUPPLIED_BY          │ sign=0 exp=6 │ NEAR-SYNONYM (same rung 2, same direction, d=115)
// d=  119 │ NIB4_NEAR            ↔ CADENCE_TRUTH        │ sign=0 exp=6 │ RUNG-SHIFT (1->2, 0 sign flips)
// d=  120 │ BERT_NEAR            ↔ CADENCE_TRUTH        │ sign=0 exp=6 │ RUNG-SHIFT (1->2, 0 sign flips)
// d=  120 │ DEPLOYED_BY          ↔ INVESTED_IN          │ sign=0 exp=6 │ RELATED (d=120, 0 sign flips)
// d=  123 │ BERT_NEAR            ↔ STRUCTURAL_TRUTH     │ sign=0 exp=6 │ NEAR-SYNONYM (same rung 1, same direction, d=123)
// d=  124 │ IS_CAUSED_BY         ↔ SUPPLIED_BY          │ sign=0 exp=7 │ NEAR-SYNONYM (same rung 2, same direction, d=124)
// d=  124 │ DEVELOPED_BY         ↔ DEPLOYED_BY          │ sign=0 exp=6 │ NEAR-SYNONYM (same rung 2, same direction, d=124)
// d=  126 │ NIB4_NEAR            ↔ STRUCTURAL_TRUTH     │ sign=0 exp=6 │ NEAR-SYNONYM (same rung 1, same direction, d=126)
// d=  130 │ STRUCTURAL_TRUTH     ↔ VALID_FOR            │ sign=0 exp=6 │ RELATED (d=130, 0 sign flips)
// d=  134 │ STRUCTURAL_TRUTH     ↔ BELONGS_TO           │ sign=0 exp=6 │ RELATED (d=134, 0 sign flips)
// d=  143 │ DEPLOYED_BY          ↔ OPERATED_BY          │ sign=0 exp=8 │ NEAR-SYNONYM (same rung 2, same direction, d=143)
// d=  156 │ NIB4_NEAR            ↔ INVESTED_IN          │ sign=0 exp=8 │ RUNG-SHIFT (1->2, 0 sign flips)
// d=  159 │ BERT_NEAR            ↔ INVESTED_IN          │ sign=0 exp=8 │ RUNG-SHIFT (1->2, 0 sign flips)
// d=  160 │ VALID_FOR            ↔ DEPLOYED_BY          │ sign=0 exp=8 │ PARTIAL-INVERSION (0 sign flips, rung 1→2)
// d=  162 │ BELONGS_TO           ↔ DEPLOYED_BY          │ sign=0 exp=8 │ PARTIAL-INVERSION (0 sign flips, rung 1→2)
// d=  162 │ SUPPLIED_BY          ↔ INVESTED_IN          │ sign=0 exp=8 │ RELATED (d=162, 0 sign flips)
// d=  166 │ TRANSFORMS           ↔ BERT_NEAR            │ sign=0 exp=9 │ RUNG-SHIFT (2->1, 0 sign flips)
// d=  169 │ TRANSFORMS           ↔ NIB4_NEAR            │ sign=0 exp=9 │ RUNG-SHIFT (2->1, 0 sign flips)
// d=  179 │ HAS_MODE             ↔ VALID_FOR            │ sign=0 exp=10 │ NEAR-SYNONYM (same rung 1, same direction, d=179)
// d=  181 │ NIB4_NEAR            ↔ OPERATED_BY          │ sign=0 exp=10 │ RUNG-SHIFT (1->2, 0 sign flips)
// d=  182 │ NIB4_NEAR            ↔ DEPLOYED_BY          │ sign=0 exp=10 │ RUNG-SHIFT (1->2, 0 sign flips)
// d=  184 │ TRANSFORMS           ↔ CADENCE_TRUTH        │ sign=0 exp=9 │ NEAR-SYNONYM (same rung 2, same direction, d=184)
// d=  184 │ BERT_NEAR            ↔ OPERATED_BY          │ sign=0 exp=10 │ RUNG-SHIFT (1->2, 0 sign flips)
// d=  184 │ DEPLOYED_BY          ↔ SUPPLIED_BY          │ sign=0 exp=10 │ NEAR-SYNONYM (same rung 2, same direction, d=184)
// d=  185 │ BERT_NEAR            ↔ DEPLOYED_BY          │ sign=0 exp=10 │ RUNG-SHIFT (1->2, 0 sign flips)
// d=  189 │ BELONGS_TO           ↔ HAS_MODE             │ sign=0 exp=10 │ NEAR-SYNONYM (same rung 1, same direction, d=189)
// d=  190 │ BELONGS_TO           ↔ DEVELOPED_BY         │ sign=0 exp=10 │ PARTIAL-INVERSION (0 sign flips, rung 1→2)
// d=  192 │ TRANSFORMS           ↔ HAS_MODE             │ sign=0 exp=11 │ RUNG-SHIFT (2->1, 0 sign flips)
// d=  196 │ IS_CAUSED_BY         ↔ DEVELOPED_BY         │ sign=0 exp=11 │ NEAR-SYNONYM (same rung 2, same direction, d=196)
// d=  197 │ CAUSES               ↔ OPERATED_BY          │ sign=0 exp=10 │ PARTIAL-INVERSION (0 sign flips, rung 2→2)
// d=  199 │ CADENCE_TRUTH        ↔ INVESTED_IN          │ sign=0 exp=10 │ NEAR-SYNONYM (same rung 2, same direction, d=199)
// d=  199 │ SURFACE_SYNONYMY     ↔ HAS_MODE             │ sign=0 exp=11 │ RELATED (d=199, 0 sign flips)
// d=  202 │ VALID_FOR            ↔ DEVELOPED_BY         │ sign=0 exp=10 │ PARTIAL-INVERSION (0 sign flips, rung 1→2)
// d=  203 │ STRUCTURAL_TRUTH     ↔ CADENCE_TRUTH        │ sign=0 exp=10 │ RUNG-SHIFT (1->2, 0 sign flips)
// d=  205 │ STRUCTURAL_TRUTH     ↔ OPERATED_BY          │ sign=0 exp=10 │ RUNG-SHIFT (1->2, 0 sign flips)
// d=  207 │ IS_CAUSED_BY         ↔ OPERATED_BY          │ sign=0 exp=11 │ SAME-CLASS-DIVERGENT (rung 2, d=207)
// d=  208 │ STRUCTURAL_TRUTH     ↔ SURFACE_SYNONYMY     │ sign=0 exp=11 │ SAME-CLASS-DIVERGENT (rung 1, d=208)
// d=  213 │ CADENCE_TRUTH        ↔ SURFACE_SYNONYMY     │ sign=0 exp=11 │ RUNG-SHIFT (2->1, 0 sign flips)
// d=  214 │ STRUCTURAL_TRUTH     ↔ INVESTED_IN          │ sign=0 exp=10 │ RUNG-SHIFT (1->2, 0 sign flips)
// d=  214 │ CADENCE_TRUTH        ↔ OPERATED_BY          │ sign=0 exp=12 │ PARTIAL-INVERSION (0 sign flips, rung 2→2)
// d=  217 │ TRANSFORMS           ↔ SURFACE_SYNONYMY     │ sign=0 exp=12 │ RUNG-SHIFT (2->1, 0 sign flips)
// d=  219 │ TRANSFORMS           ↔ INVESTED_IN          │ sign=0 exp=11 │ SAME-CLASS-DIVERGENT (rung 2, d=219)
// d=  220 │ CAUSES               ↔ DEVELOPED_BY         │ sign=0 exp=12 │ PARTIAL-INVERSION (0 sign flips, rung 2→2)
// d=  220 │ NIB4_NEAR            ↔ DEVELOPED_BY         │ sign=0 exp=12 │ RUNG-SHIFT (1->2, 0 sign flips)
// d=  221 │ BERT_NEAR            ↔ VALID_FOR            │ sign=0 exp=12 │ RELATED (d=221, 0 sign flips)
// d=  222 │ NIB4_NEAR            ↔ VALID_FOR            │ sign=0 exp=12 │ RELATED (d=222, 0 sign flips)
// d=  223 │ BERT_NEAR            ↔ DEVELOPED_BY         │ sign=0 exp=12 │ RUNG-SHIFT (1->2, 0 sign flips)
// d=  223 │ CADENCE_TRUTH        ↔ DEPLOYED_BY          │ sign=0 exp=12 │ PARTIAL-INVERSION (0 sign flips, rung 2→2)
// d=  224 │ NIB4_NEAR            ↔ BELONGS_TO           │ sign=0 exp=12 │ RELATED (d=224, 0 sign flips)
// d=  224 │ HAS_MODE             ↔ OPERATED_BY          │ sign=0 exp=12 │ PARTIAL-INVERSION (0 sign flips, rung 1→2)
// d=  226 │ BELONGS_TO           ↔ INVESTED_IN          │ sign=0 exp=12 │ RUNG-SHIFT (1->2, 0 sign flips)
// d=  227 │ BERT_NEAR            ↔ BELONGS_TO           │ sign=0 exp=12 │ RELATED (d=227, 0 sign flips)
// d=  228 │ CADENCE_TRUTH        ↔ HAS_MODE             │ sign=0 exp=12 │ RUNG-SHIFT (2->1, 0 sign flips)
// d=  231 │ BELONGS_TO           ↔ OPERATED_BY          │ sign=0 exp=12 │ PARTIAL-INVERSION (0 sign flips, rung 1→2)
// d=  231 │ HAS_MODE             ↔ INVESTED_IN          │ sign=0 exp=12 │ RUNG-SHIFT (1->2, 0 sign flips)
// d=  231 │ VALID_FOR            ↔ OPERATED_BY          │ sign=0 exp=12 │ PARTIAL-INVERSION (0 sign flips, rung 1→2)
// d=  232 │ VALID_FOR            ↔ INVESTED_IN          │ sign=0 exp=12 │ RUNG-SHIFT (1->2, 0 sign flips)
// d=  234 │ CAUSES               ↔ INVESTED_IN          │ sign=0 exp=12 │ SAME-CLASS-DIVERGENT (rung 2, d=234)
// d=  236 │ CAUSES               ↔ TARGETS              │ sign=0 exp=13 │ RUNG-SHIFT (2->3, 0 sign flips)
// d=  237 │ CAUSES               ↔ HAS_MODE             │ sign=0 exp=12 │ RUNG-SHIFT (2->1, 0 sign flips)
// d=  241 │ SURFACE_SYNONYMY     ↔ OPERATED_BY          │ sign=0 exp=13 │ RUNG-SHIFT (1->2, 0 sign flips)
// d=  244 │ TRANSFORMS           ↔ OPERATED_BY          │ sign=0 exp=13 │ PARTIAL-INVERSION (0 sign flips, rung 2→2)
// d=  244 │ STRUCTURAL_TRUTH     ↔ DEVELOPED_BY         │ sign=0 exp=12 │ RUNG-SHIFT (1->2, 0 sign flips)
// d=  245 │ DISSOLVES_INTO       ↔ STRUCTURAL_TRUTH     │ sign=0 exp=13 │ RUNG-SHIFT (3->1, 0 sign flips)
// d=  246 │ CAUSES               ↔ STRUCTURAL_TRUTH     │ sign=0 exp=12 │ RUNG-SHIFT (2->1, 0 sign flips)
// d=  248 │ NIB4_NEAR            ↔ TARGETS              │ sign=0 exp=13 │ RUNG-SHIFT (1->3, 0 sign flips)
// d=  249 │ TRANSFORMS           ↔ DEPLOYED_BY          │ sign=0 exp=13 │ PARTIAL-INVERSION (0 sign flips, rung 2→2)
// d=  250 │ IS_CAUSED_BY         ↔ INVESTED_IN          │ sign=0 exp=13 │ RELATED (d=250, 0 sign flips)
// d=  251 │ BERT_NEAR            ↔ TARGETS              │ sign=0 exp=13 │ RUNG-SHIFT (1->3, 0 sign flips)
// d=  252 │ SURFACE_SYNONYMY     ↔ INVESTED_IN          │ sign=0 exp=13 │ RUNG-SHIFT (1->2, 0 sign flips)
// d=  257 │ CADENCE_TRUTH        ↔ DEVELOPED_BY         │ sign=0 exp=14 │ PARTIAL-INVERSION (0 sign flips, rung 2→2)
// d=  261 │ HAS_MODE             ↔ DEVELOPED_BY         │ sign=0 exp=14 │ PARTIAL-INVERSION (0 sign flips, rung 1→2)
// d=  262 │ CAUSES               ↔ NIB4_NEAR            │ sign=0 exp=14 │ RUNG-SHIFT (2->1, 0 sign flips)
// d=  264 │ CAUSES               ↔ IS_CAUSED_BY         │ sign=0 exp=15 │ PARTIAL-INVERSION (0 sign flips, rung 2→2)
// d=  265 │ CAUSES               ↔ BERT_NEAR            │ sign=0 exp=14 │ RUNG-SHIFT (2->1, 0 sign flips)
// d=  265 │ CAUSES               ↔ CADENCE_TRUTH        │ sign=0 exp=14 │ SAME-CLASS-DIVERGENT (rung 2, d=265)
// d=  265 │ TRANSFORMS           ↔ STRUCTURAL_TRUTH     │ sign=0 exp=15 │ RUNG-SHIFT (2->1, 0 sign flips)
// d=  270 │ BELONGS_TO           ↔ SUPPLIED_BY          │ sign=0 exp=14 │ PARTIAL-INVERSION (0 sign flips, rung 1→2)
// d=  272 │ VALID_FOR            ↔ SUPPLIED_BY          │ sign=0 exp=14 │ PARTIAL-INVERSION (0 sign flips, rung 1→2)
// d=  274 │ STRUCTURAL_TRUTH     ↔ DEPLOYED_BY          │ sign=0 exp=14 │ RUNG-SHIFT (1->2, 0 sign flips)
// d=  274 │ SURFACE_SYNONYMY     ↔ TARGETS              │ sign=0 exp=14 │ RUNG-SHIFT (1->3, 0 sign flips)
// d=  275 │ HAS_MODE             ↔ TARGETS              │ sign=0 exp=15 │ RUNG-SHIFT (1->3, 0 sign flips)
// d=  276 │ IS_CAUSED_BY         ↔ DEPLOYED_BY          │ sign=0 exp=15 │ SAME-CLASS-DIVERGENT (rung 2, d=276)
// d=  276 │ SURFACE_SYNONYMY     ↔ DEVELOPED_BY         │ sign=0 exp=15 │ RUNG-SHIFT (1->2, 0 sign flips)
// d=  281 │ TRANSFORMS           ↔ DEVELOPED_BY         │ sign=0 exp=15 │ PARTIAL-INVERSION (0 sign flips, rung 2→2)
// d=  282 │ SURFACE_SYNONYMY     ↔ DEPLOYED_BY          │ sign=0 exp=15 │ RUNG-SHIFT (1->2, 0 sign flips)
// d=  283 │ DISSOLVES_INTO       ↔ BELONGS_TO           │ sign=0 exp=15 │ RUNG-SHIFT (3->1, 0 sign flips)
// d=  285 │ IS_CAUSED_BY         ↔ HAS_MODE             │ sign=0 exp=15 │ RUNG-SHIFT (2->1, 0 sign flips)
// d=  288 │ CAUSES               ↔ SUPPLIED_BY          │ sign=0 exp=16 │ PARTIAL-INVERSION (0 sign flips, rung 2→2)
// d=  289 │ DISSOLVES_INTO       ↔ VALID_FOR            │ sign=0 exp=15 │ RUNG-SHIFT (3->1, 0 sign flips)
// d=  289 │ CADENCE_TRUTH        ↔ TARGETS              │ sign=0 exp=15 │ RUNG-SHIFT (2->3, 0 sign flips)
// d=  290 │ IS_CAUSED_BY         ↔ BELONGS_TO           │ sign=0 exp=15 │ RUNG-SHIFT (2->1, 0 sign flips)
// d=  290 │ NIB4_NEAR            ↔ SUPPLIED_BY          │ sign=0 exp=16 │ RUNG-SHIFT (1->2, 0 sign flips)
// d=  293 │ BERT_NEAR            ↔ SUPPLIED_BY          │ sign=0 exp=16 │ RUNG-SHIFT (1->2, 0 sign flips)
// d=  293 │ HAS_MODE             ↔ DEPLOYED_BY          │ sign=0 exp=16 │ PARTIAL-INVERSION (0 sign flips, rung 1→2)
// d=  295 │ CADENCE_TRUTH        ↔ VALID_FOR            │ sign=0 exp=16 │ RUNG-SHIFT (2->1, 0 sign flips)
// d=  296 │ IS_CAUSED_BY         ↔ VALID_FOR            │ sign=0 exp=15 │ RUNG-SHIFT (2->1, 0 sign flips)
// d=  297 │ TRANSFORMS           ↔ TARGETS              │ sign=0 exp=16 │ RUNG-SHIFT (2->3, 0 sign flips)
// d=  302 │ CAUSES               ↔ BELONGS_TO           │ sign=0 exp=16 │ RUNG-SHIFT (2->1, 0 sign flips)
// d=  302 │ SURFACE_SYNONYMY     ↔ BELONGS_TO           │ sign=0 exp=17 │ RELATED (d=302, 0 sign flips)
// d=  305 │ CADENCE_TRUTH        ↔ BELONGS_TO           │ sign=0 exp=16 │ RUNG-SHIFT (2->1, 0 sign flips)
// d=  306 │ STRUCTURAL_TRUTH     ↔ SUPPLIED_BY          │ sign=0 exp=16 │ RUNG-SHIFT (1->2, 0 sign flips)
// d=  308 │ SURFACE_SYNONYMY     ↔ VALID_FOR            │ sign=0 exp=17 │ RELATED (d=308, 0 sign flips)
// d=  310 │ CAUSES               ↔ VALID_FOR            │ sign=0 exp=16 │ RUNG-SHIFT (2->1, 0 sign flips)
// d=  310 │ IS_CAUSED_BY         ↔ NIB4_NEAR            │ sign=0 exp=17 │ RUNG-SHIFT (2->1, 0 sign flips)
// d=  313 │ CAUSES               ↔ TRANSFORMS           │ sign=0 exp=17 │ SAME-CLASS-DIVERGENT (rung 2, d=313)
// d=  313 │ IS_CAUSED_BY         ↔ BERT_NEAR            │ sign=0 exp=17 │ RUNG-SHIFT (2->1, 0 sign flips)
// d=  318 │ CAUSES               ↔ SURFACE_SYNONYMY     │ sign=0 exp=17 │ RUNG-SHIFT (2->1, 0 sign flips)
// d=  319 │ CADENCE_TRUTH        ↔ SUPPLIED_BY          │ sign=0 exp=18 │ PARTIAL-INVERSION (0 sign flips, rung 2→2)
// d=  322 │ IS_CAUSED_BY         ↔ TARGETS              │ sign=0 exp=18 │ RUNG-SHIFT (2->3, 0 sign flips)
// d=  322 │ DISSOLVES_INTO       ↔ HAS_MODE             │ sign=0 exp=17 │ RUNG-SHIFT (3->1, 0 sign flips)
// d=  326 │ IS_CAUSED_BY         ↔ STRUCTURAL_TRUTH     │ sign=0 exp=17 │ RUNG-SHIFT (2->1, 0 sign flips)
// d=  326 │ INVESTED_IN          ↔ TARGETS              │ sign=0 exp=17 │ RUNG-SHIFT (2->3, 0 sign flips)
// d=  329 │ HAS_MODE             ↔ SUPPLIED_BY          │ sign=0 exp=18 │ PARTIAL-INVERSION (0 sign flips, rung 1→2)
// d=  334 │ TRANSFORMS           ↔ DISSOLVES_INTO       │ sign=0 exp=18 │ RUNG-SHIFT (2->3, 0 sign flips)
// d=  338 │ CAUSES               ↔ DEPLOYED_BY          │ sign=0 exp=18 │ PARTIAL-INVERSION (0 sign flips, rung 2→2)
// d=  338 │ SUPPLIED_BY          ↔ TARGETS              │ sign=0 exp=19 │ RUNG-SHIFT (2->3, 0 sign flips)
// d=  339 │ IS_CAUSED_BY         ↔ CADENCE_TRUTH        │ sign=0 exp=19 │ RELATED (d=339, 0 sign flips)
// d=  340 │ STRUCTURAL_TRUTH     ↔ TARGETS              │ sign=0 exp=19 │ RUNG-SHIFT (1->3, 0 sign flips)
// d=  349 │ TRANSFORMS           ↔ SUPPLIED_BY          │ sign=0 exp=19 │ PARTIAL-INVERSION (0 sign flips, rung 2→2)
// d=  349 │ OPERATED_BY          ↔ TARGETS              │ sign=0 exp=19 │ RUNG-SHIFT (2->3, 0 sign flips)
// d=  352 │ SURFACE_SYNONYMY     ↔ SUPPLIED_BY          │ sign=0 exp=19 │ RUNG-SHIFT (1->2, 0 sign flips)
// d=  353 │ DISSOLVES_INTO       ↔ NIB4_NEAR            │ sign=0 exp=19 │ RUNG-SHIFT (3->1, 0 sign flips)
// d=  354 │ DISSOLVES_INTO       ↔ BERT_NEAR            │ sign=0 exp=19 │ RUNG-SHIFT (3->1, 0 sign flips)
// d=  354 │ DEPLOYED_BY          ↔ TARGETS              │ sign=0 exp=19 │ RUNG-SHIFT (2->3, 0 sign flips)
// d=  356 │ DISSOLVES_INTO       ↔ OPERATED_BY          │ sign=0 exp=19 │ PARTIAL-INVERSION (0 sign flips, rung 3→2)
// d=  357 │ TRANSFORMS           ↔ BELONGS_TO           │ sign=0 exp=21 │ RUNG-SHIFT (2->1, 0 sign flips)
// d=  359 │ CAUSES               ↔ DISSOLVES_INTO       │ sign=0 exp=19 │ RUNG-SHIFT (2->3, 0 sign flips)
// d=  359 │ DISSOLVES_INTO       ↔ DEPLOYED_BY          │ sign=0 exp=19 │ PARTIAL-INVERSION (0 sign flips, rung 3→2)
// d=  359 │ DISSOLVES_INTO       ↔ INVESTED_IN          │ sign=0 exp=19 │ RUNG-SHIFT (3->2, 0 sign flips)
// d=  364 │ DISSOLVES_INTO       ↔ CADENCE_TRUTH        │ sign=0 exp=19 │ RUNG-SHIFT (3->2, 0 sign flips)
// d=  365 │ TRANSFORMS           ↔ VALID_FOR            │ sign=0 exp=21 │ RUNG-SHIFT (2->1, 0 sign flips)
// d=  368 │ IS_CAUSED_BY         ↔ SURFACE_SYNONYMY     │ sign=0 exp=20 │ RUNG-SHIFT (2->1, 0 sign flips)
// d=  369 │ IS_CAUSED_BY         ↔ TRANSFORMS           │ sign=0 exp=20 │ RELATED (d=369, 0 sign flips)
// d=  369 │ DISSOLVES_INTO       ↔ SURFACE_SYNONYMY     │ sign=0 exp=20 │ RUNG-SHIFT (3->1, 0 sign flips)
// d=  379 │ DISSOLVES_INTO       ↔ DEVELOPED_BY         │ sign=0 exp=21 │ PARTIAL-INVERSION (0 sign flips, rung 3→2)
// d=  380 │ DEVELOPED_BY         ↔ TARGETS              │ sign=0 exp=21 │ RUNG-SHIFT (2->3, 0 sign flips)
// d=  418 │ BELONGS_TO           ↔ TARGETS              │ sign=0 exp=23 │ RUNG-SHIFT (1->3, 0 sign flips)
// d=  418 │ VALID_FOR            ↔ TARGETS              │ sign=0 exp=23 │ RUNG-SHIFT (1->3, 0 sign flips)
// d=  453 │ DISSOLVES_INTO       ↔ SUPPLIED_BY          │ sign=0 exp=25 │ PARTIAL-INVERSION (0 sign flips, rung 3→2)
// d=  469 │ IS_CAUSED_BY         ↔ DISSOLVES_INTO       │ sign=0 exp=26 │ RUNG-SHIFT (2->3, 0 sign flips)
// d=  503 │ DISSOLVES_INTO       ↔ TARGETS              │ sign=0 exp=28 │ SAME-CLASS-DIVERGENT (rung 3, d=503)


// ═══ SECTION 5: CANONICAL CYPHER PATTERNS ═══
// Battle-tested query patterns from corpus.

// --- Pattern 1: Create qualia node with Nib4 fingerprint ---
MERGE (n:QualiaItem {id: $id, label: $label, family: $family,
                      mode: $mode, tau: $tau, nib4: $nib4})

// --- Pattern 2: Structural truth edge (both Nib4 and BERT agree) ---
MERGE (a)-[:STRUCTURAL_TRUTH {nib4d: $nib4_dist, bertd: $bert_dist}]->(b)

// --- Pattern 3: Cadence truth edge (Nib4 close, BERT far) ---
MERGE (a)-[:CADENCE_TRUTH {nib4d: $nib4_dist, bertd: $bert_dist}]->(b)

// --- Pattern 4: Surface synonymy edge (BERT close, Nib4 far) ---
MERGE (a)-[:SURFACE_SYNONYMY {nib4d: $nib4_dist, bertd: $bert_dist}]->(b)

// --- Pattern 5: Causal edge with voice direction ---
MERGE (a)-[:CAUSES {agency: $agency, volition: $volition}]->(b)
MERGE (b)-[:IS_CAUSED_BY {gravity: $gravity, resonance: $resonance}]->(a)

// --- Pattern 6: Transformation edge ---
MERGE (a)-[:TRANSFORMS {dissonance: $dissonance, friction: $friction}]->(b)

// --- Pattern 7: Dissolution edge (ecstatic collapse) ---
MERGE (a)-[:DISSOLVES_INTO {resonance: $resonance, staunen: $staunen}]->(b)

// --- Pattern 8: AIWar system node ---
MERGE (s:System {id: $id, name: $name, year: $year,
                  system_type: $type, ml_task: $ml_task})

// --- Pattern 9: AIWar stakeholder with AIRO type ---
MERGE (st:Stakeholder {id: $id, name: $name,
                        stakeholder_type: $type, airo_type: $airo})

// --- Pattern 10: Schema axis with valid values ---
MERGE (a:SchemaAxis {name: $axis_name})
MERGE (v:SchemaValue {name: $value_name})
MERGE (v)-[:VALID_FOR]->(a)

// --- Query: Find structural truths within family ---
MATCH (a:QualiaItem)-[:STRUCTURAL_TRUTH]-(b:QualiaItem)
WHERE a.family = $family AND b.family = $family
RETURN a.label, b.label, a.tau, b.tau
ORDER BY a.tau DESC

// --- Query: Surface synonymy traps (BERT confused, Nib4 correct) ---
MATCH (a:QualiaItem)-[r:SURFACE_SYNONYMY]-(b:QualiaItem)
WHERE r.nib4d > 100
RETURN a.label, b.label, r.nib4d, r.bertd
ORDER BY r.nib4d DESC
LIMIT 20

// --- Query: Causal chain traversal ---
MATCH path = (a:QualiaItem)-[:CAUSES*1..3]->(b:QualiaItem)
WHERE a.id = $start_id
RETURN path

// --- Query: Voice inversion detection ---
MATCH (a)-[r1:CAUSES]->(b), (b)-[r2:IS_CAUSED_BY]->(a)
RETURN a.label, b.label, r1.agency, r2.gravity

// --- Query: Cross-domain bridge (qualia ↔ aiwar) ---
MATCH (q:QualiaItem)-[:STRUCTURAL_TRUTH]-(q2:QualiaItem)
MATCH (s:System)-[:DEVELOPED_BY]->(st:Stakeholder)
WHERE q.family = 'Power' AND st.stakeholder_type = 'Nation'
RETURN q.label, s.name, st.name

// ═══ END OF CONTRACT ═══
