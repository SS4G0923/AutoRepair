-- 2026-04-18_add_benchmark_real_repair_and_experiments.sql
-- Extends the benchmark subsystem with:
--   1) real-repair bookkeeping on benchmark_runs
--      (is_plausible / is_correct / strategy / token usage / patch stats)
--   2) a benchmark_experiments table to group runs into comparison studies
--      ("weak model + our system" vs "strong model raw", ablations, etc.)
--   3) a leaderboard-style aggregation view keyed by experiment

-- --------------------------------------------------------------------------
-- 1. Extend benchmark_runs with repair-evaluation & experiment fields
--    (the migration runner swallows duplicate-column errors 1060/1061, so
--     each ALTER is safe to re-run on an already-migrated database.)
-- --------------------------------------------------------------------------

ALTER TABLE benchmark_runs
    ADD COLUMN is_plausible TINYINT(1) NOT NULL DEFAULT 0
        COMMENT 'Patch applied, project recompiled, and all defects4j tests pass.';
ALTER TABLE benchmark_runs
    ADD COLUMN is_correct TINYINT(1) NOT NULL DEFAULT 0
        COMMENT 'Every originally-failing trigger test now passes and no new failures introduced.';
ALTER TABLE benchmark_runs
    ADD COLUMN strategy VARCHAR(48) NOT NULL DEFAULT 'inspect_only'
        COMMENT 'Repair strategy: inspect_only | naive_chat | full_pipeline | no_ranking | ...';
ALTER TABLE benchmark_runs
    ADD COLUMN experiment_id INT DEFAULT NULL
        COMMENT 'Groups runs into a comparison experiment (NULL for ad-hoc runs).';
ALTER TABLE benchmark_runs
    ADD COLUMN prompt_tokens INT NOT NULL DEFAULT 0;
ALTER TABLE benchmark_runs
    ADD COLUMN completion_tokens INT NOT NULL DEFAULT 0;
ALTER TABLE benchmark_runs
    ADD COLUMN total_tokens INT NOT NULL DEFAULT 0;
ALTER TABLE benchmark_runs
    ADD COLUMN patch_lines_added INT NOT NULL DEFAULT 0;
ALTER TABLE benchmark_runs
    ADD COLUMN patch_lines_removed INT NOT NULL DEFAULT 0;
ALTER TABLE benchmark_runs
    ADD COLUMN llm_rounds SMALLINT NOT NULL DEFAULT 0
        COMMENT 'How many LLM calls were issued while attempting to repair.';
ALTER TABLE benchmark_runs
    ADD COLUMN failed_tests_before INT NOT NULL DEFAULT 0;
ALTER TABLE benchmark_runs
    ADD COLUMN failed_tests_after INT NOT NULL DEFAULT 0;
ALTER TABLE benchmark_runs
    ADD KEY idx_benchmark_runs_experiment (experiment_id);
ALTER TABLE benchmark_runs
    ADD KEY idx_benchmark_runs_strategy (strategy);

-- --------------------------------------------------------------------------
-- 2. benchmark_experiments – a "study" grouping many runs together
-- --------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS benchmark_experiments (
    id                   INT AUTO_INCREMENT PRIMARY KEY,
    experiment_code      VARCHAR(80)  NOT NULL UNIQUE
        COMMENT 'Short slug used by the CLI (e.g. `weak-vs-strong-v1`).',
    title                VARCHAR(160) NOT NULL,
    description          TEXT,
    hypothesis           TEXT
        COMMENT 'Free-form hypothesis, e.g. "weak + scaffolding beats strong raw".',
    created_by_user_id   INT,
    status               VARCHAR(24) NOT NULL DEFAULT 'pending'
        COMMENT 'pending | running | completed | cancelled | failed',
    config_json          MEDIUMTEXT
        COMMENT 'JSON snapshot of experiment config (models, bugs, strategies).',
    created_at           TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    started_at           TIMESTAMP NULL,
    finished_at          TIMESTAMP NULL,
    total_runs           INT NOT NULL DEFAULT 0,
    completed_runs       INT NOT NULL DEFAULT 0,
    failed_runs          INT NOT NULL DEFAULT 0,
    KEY idx_experiments_status (status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- --------------------------------------------------------------------------
-- 3. Seed a "canonical" experiment slot used by the default runner script.
--    (Additional experiments can be created on the fly from the CLI.)
-- --------------------------------------------------------------------------

INSERT INTO benchmark_experiments
    (experiment_code, title, description, hypothesis, status)
SELECT
    'weak-vs-strong-v1',
    'Weak+Pipeline vs Strong+Raw',
    'Shared Defects4J bug set benchmarked across two arms: (A) a strong baseline LLM driven by a naive chat prompt, and (B) a weaker LLM driven by the AutoRepair evidence pipeline.',
    'A weak model combined with evidence-extraction + suspect-ranking will achieve a higher plausible/correct rate than a strong model given only raw buggy code.',
    'pending'
WHERE NOT EXISTS (
    SELECT 1 FROM benchmark_experiments WHERE experiment_code = 'weak-vs-strong-v1'
);
