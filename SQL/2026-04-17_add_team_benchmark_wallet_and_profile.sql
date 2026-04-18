USE `AutoRepair`;

-- ---------------------------------------------------------------------------
-- Module A: Team / Organization & Project
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS `organizations` (
    `id` BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    `name` VARCHAR(120) NOT NULL,
    `slug` VARCHAR(64) NOT NULL,
    `description` TEXT NULL,
    `owner_user_id` BIGINT UNSIGNED NOT NULL,
    `plan_code` VARCHAR(32) NOT NULL DEFAULT 'free_team',
    `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    `updated_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    `deleted_at` DATETIME NULL DEFAULT NULL,
    PRIMARY KEY (`id`),
    UNIQUE KEY `uk_organizations_slug` (`slug`),
    KEY `idx_organizations_owner` (`owner_user_id`, `deleted_at`),
    CONSTRAINT `fk_organizations_owner_user_id`
        FOREIGN KEY (`owner_user_id`) REFERENCES `users` (`id`)
        ON DELETE CASCADE
        ON UPDATE CASCADE
);

CREATE TABLE IF NOT EXISTS `organization_members` (
    `id` BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    `organization_id` BIGINT UNSIGNED NOT NULL,
    `user_id` BIGINT UNSIGNED NOT NULL,
    `member_role` VARCHAR(16) NOT NULL DEFAULT 'member',
    `joined_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (`id`),
    UNIQUE KEY `uk_organization_members_org_user` (`organization_id`, `user_id`),
    KEY `idx_organization_members_user` (`user_id`),
    CONSTRAINT `fk_organization_members_organization_id`
        FOREIGN KEY (`organization_id`) REFERENCES `organizations` (`id`)
        ON DELETE CASCADE
        ON UPDATE CASCADE,
    CONSTRAINT `fk_organization_members_user_id`
        FOREIGN KEY (`user_id`) REFERENCES `users` (`id`)
        ON DELETE CASCADE
        ON UPDATE CASCADE
);

CREATE TABLE IF NOT EXISTS `organization_invites` (
    `id` BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    `organization_id` BIGINT UNSIGNED NOT NULL,
    `email` VARCHAR(255) NOT NULL,
    `invite_token` VARCHAR(64) NOT NULL,
    `invited_by_user_id` BIGINT UNSIGNED NULL,
    `invite_status` VARCHAR(16) NOT NULL DEFAULT 'pending',
    `expires_at` DATETIME NOT NULL,
    `accepted_at` DATETIME NULL DEFAULT NULL,
    `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (`id`),
    UNIQUE KEY `uk_organization_invites_token` (`invite_token`),
    KEY `idx_organization_invites_org_status` (`organization_id`, `invite_status`),
    CONSTRAINT `fk_organization_invites_organization_id`
        FOREIGN KEY (`organization_id`) REFERENCES `organizations` (`id`)
        ON DELETE CASCADE
        ON UPDATE CASCADE,
    CONSTRAINT `fk_organization_invites_invited_by_user_id`
        FOREIGN KEY (`invited_by_user_id`) REFERENCES `users` (`id`)
        ON DELETE SET NULL
        ON UPDATE CASCADE
);

CREATE TABLE IF NOT EXISTS `projects` (
    `id` BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    `organization_id` BIGINT UNSIGNED NOT NULL,
    `owner_user_id` BIGINT UNSIGNED NOT NULL,
    `name` VARCHAR(120) NOT NULL,
    `slug` VARCHAR(64) NOT NULL,
    `language` VARCHAR(24) NULL,
    `description` TEXT NULL,
    `repo_url` VARCHAR(1024) NULL,
    `default_entrypoint` VARCHAR(255) NULL,
    `color_hex` VARCHAR(16) NULL,
    `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    `updated_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    `deleted_at` DATETIME NULL DEFAULT NULL,
    PRIMARY KEY (`id`),
    UNIQUE KEY `uk_projects_org_slug` (`organization_id`, `slug`),
    KEY `idx_projects_owner` (`owner_user_id`),
    KEY `idx_projects_org_updated` (`organization_id`, `updated_at`),
    CONSTRAINT `fk_projects_organization_id`
        FOREIGN KEY (`organization_id`) REFERENCES `organizations` (`id`)
        ON DELETE CASCADE
        ON UPDATE CASCADE,
    CONSTRAINT `fk_projects_owner_user_id`
        FOREIGN KEY (`owner_user_id`) REFERENCES `users` (`id`)
        ON DELETE CASCADE
        ON UPDATE CASCADE
);

-- Attach conversation histories to a project (optional).
-- NOTE: statements below are split and the migration runner silently ignores
-- duplicate-column (1060) / duplicate-key (1061) errors, keeping the migration idempotent.
ALTER TABLE `conversation_histories` ADD COLUMN `project_id` BIGINT UNSIGNED NULL AFTER `user_id`;
ALTER TABLE `conversation_histories` ADD KEY `idx_conversation_histories_project_updated` (`project_id`, `updated_at`);

-- ---------------------------------------------------------------------------
-- Module D: Credit Wallet
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS `credit_wallets` (
    `user_id` BIGINT UNSIGNED NOT NULL,
    `balance_credits` INT NOT NULL DEFAULT 0,
    `lifetime_earned` INT NOT NULL DEFAULT 0,
    `lifetime_spent` INT NOT NULL DEFAULT 0,
    `last_grant_at` DATETIME NULL DEFAULT NULL,
    `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    `updated_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (`user_id`),
    CONSTRAINT `fk_credit_wallets_user_id`
        FOREIGN KEY (`user_id`) REFERENCES `users` (`id`)
        ON DELETE CASCADE
        ON UPDATE CASCADE
);

CREATE TABLE IF NOT EXISTS `credit_transactions` (
    `id` BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    `user_id` BIGINT UNSIGNED NOT NULL,
    `change_credits` INT NOT NULL,
    `balance_after` INT NOT NULL,
    `reason_code` VARCHAR(32) NOT NULL,
    `reference_type` VARCHAR(32) NULL,
    `reference_id` BIGINT UNSIGNED NULL,
    `note` VARCHAR(255) NULL,
    `actor_user_id` BIGINT UNSIGNED NULL,
    `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (`id`),
    KEY `idx_credit_transactions_user_created` (`user_id`, `created_at`),
    KEY `idx_credit_transactions_reason_created` (`reason_code`, `created_at`),
    CONSTRAINT `fk_credit_transactions_user_id`
        FOREIGN KEY (`user_id`) REFERENCES `users` (`id`)
        ON DELETE CASCADE
        ON UPDATE CASCADE,
    CONSTRAINT `fk_credit_transactions_actor_user_id`
        FOREIGN KEY (`actor_user_id`) REFERENCES `users` (`id`)
        ON DELETE SET NULL
        ON UPDATE CASCADE
);

CREATE TABLE IF NOT EXISTS `credit_pricing_rules` (
    `id` BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    `role_code` VARCHAR(16) NOT NULL,
    `monthly_free_credits` INT NOT NULL DEFAULT 0,
    `cost_per_chat` INT NOT NULL DEFAULT 0,
    `cost_per_repair` INT NOT NULL DEFAULT 0,
    `cost_per_benchmark_run` INT NOT NULL DEFAULT 0,
    `updated_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (`id`),
    UNIQUE KEY `uk_credit_pricing_rules_role` (`role_code`)
);

INSERT INTO `credit_pricing_rules`
    (`role_code`, `monthly_free_credits`, `cost_per_chat`, `cost_per_repair`, `cost_per_benchmark_run`)
VALUES
    ('basic',    100,  1, 10, 20),
    ('advanced', 500,  1,  5, 10),
    ('admin',  10000,  0,  0,  0)
ON DUPLICATE KEY UPDATE
    `monthly_free_credits` = VALUES(`monthly_free_credits`),
    `cost_per_chat`        = VALUES(`cost_per_chat`),
    `cost_per_repair`      = VALUES(`cost_per_repair`),
    `cost_per_benchmark_run` = VALUES(`cost_per_benchmark_run`);

-- ---------------------------------------------------------------------------
-- Module E: Benchmark (Defects4J-first)
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS `benchmark_projects` (
    `id` BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    `project_code` VARCHAR(64) NOT NULL,
    `display_name` VARCHAR(120) NOT NULL,
    `source_type` VARCHAR(32) NOT NULL DEFAULT 'defects4j',
    `language` VARCHAR(24) NOT NULL DEFAULT 'java',
    `description` TEXT NULL,
    `tags` VARCHAR(255) NULL,
    `is_active` TINYINT(1) NOT NULL DEFAULT 1,
    `sort_order` INT NOT NULL DEFAULT 0,
    `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    `updated_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (`id`),
    UNIQUE KEY `uk_benchmark_projects_code` (`project_code`),
    KEY `idx_benchmark_projects_active_sort` (`is_active`, `sort_order`)
);

CREATE TABLE IF NOT EXISTS `benchmark_bugs` (
    `id` BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    `project_id` BIGINT UNSIGNED NOT NULL,
    `bug_key` VARCHAR(64) NOT NULL,
    `title` VARCHAR(255) NOT NULL,
    `severity` VARCHAR(16) NOT NULL DEFAULT 'normal',
    `defects4j_project` VARCHAR(32) NULL,
    `defects4j_bug_id` INT NULL,
    `description` TEXT NULL,
    `tags` VARCHAR(255) NULL,
    `is_active` TINYINT(1) NOT NULL DEFAULT 1,
    `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    `updated_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (`id`),
    UNIQUE KEY `uk_benchmark_bugs_project_key` (`project_id`, `bug_key`),
    KEY `idx_benchmark_bugs_active` (`is_active`),
    CONSTRAINT `fk_benchmark_bugs_project_id`
        FOREIGN KEY (`project_id`) REFERENCES `benchmark_projects` (`id`)
        ON DELETE CASCADE
        ON UPDATE CASCADE
);

CREATE TABLE IF NOT EXISTS `benchmark_runs` (
    `id` BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    `user_id` BIGINT UNSIGNED NOT NULL,
    `organization_id` BIGINT UNSIGNED NULL,
    `project_id` BIGINT UNSIGNED NOT NULL,
    `bug_id` BIGINT UNSIGNED NOT NULL,
    `model_key` VARCHAR(128) NOT NULL,
    `run_mode` VARCHAR(24) NOT NULL DEFAULT 'inspect_only',
    `run_status` VARCHAR(24) NOT NULL DEFAULT 'pending',
    `stage` VARCHAR(32) NULL,
    `pass_count` INT NOT NULL DEFAULT 0,
    `fail_count` INT NOT NULL DEFAULT 0,
    `total_tests` INT NOT NULL DEFAULT 0,
    `duration_ms` INT NOT NULL DEFAULT 0,
    `credits_spent` INT NOT NULL DEFAULT 0,
    `error_message` TEXT NULL,
    `report_json` LONGTEXT NULL,
    `patch_diff` LONGTEXT NULL,
    `started_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    `finished_at` DATETIME NULL DEFAULT NULL,
    PRIMARY KEY (`id`),
    KEY `idx_benchmark_runs_user_started` (`user_id`, `started_at`),
    KEY `idx_benchmark_runs_bug_status` (`bug_id`, `run_status`),
    KEY `idx_benchmark_runs_project_model` (`project_id`, `model_key`, `run_status`),
    CONSTRAINT `fk_benchmark_runs_user_id`
        FOREIGN KEY (`user_id`) REFERENCES `users` (`id`)
        ON DELETE CASCADE
        ON UPDATE CASCADE,
    CONSTRAINT `fk_benchmark_runs_project_id`
        FOREIGN KEY (`project_id`) REFERENCES `benchmark_projects` (`id`)
        ON DELETE CASCADE
        ON UPDATE CASCADE,
    CONSTRAINT `fk_benchmark_runs_bug_id`
        FOREIGN KEY (`bug_id`) REFERENCES `benchmark_bugs` (`id`)
        ON DELETE CASCADE
        ON UPDATE CASCADE,
    CONSTRAINT `fk_benchmark_runs_organization_id`
        FOREIGN KEY (`organization_id`) REFERENCES `organizations` (`id`)
        ON DELETE SET NULL
        ON UPDATE CASCADE
);

CREATE TABLE IF NOT EXISTS `benchmark_leaderboard` (
    `id` BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    `project_id` BIGINT UNSIGNED NOT NULL,
    `model_key` VARCHAR(128) NOT NULL,
    `sample_count` INT NOT NULL DEFAULT 0,
    `success_count` INT NOT NULL DEFAULT 0,
    `pass_rate` DECIMAL(6,4) NOT NULL DEFAULT 0.0000,
    `avg_duration_ms` INT NOT NULL DEFAULT 0,
    `last_run_at` DATETIME NULL,
    `updated_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (`id`),
    UNIQUE KEY `uk_benchmark_leaderboard_project_model` (`project_id`, `model_key`),
    KEY `idx_benchmark_leaderboard_project_rate` (`project_id`, `pass_rate`),
    CONSTRAINT `fk_benchmark_leaderboard_project_id`
        FOREIGN KEY (`project_id`) REFERENCES `benchmark_projects` (`id`)
        ON DELETE CASCADE
        ON UPDATE CASCADE
);

-- Seed common Defects4J projects for convenience.
INSERT INTO `benchmark_projects` (`project_code`, `display_name`, `source_type`, `language`, `description`, `tags`, `sort_order`)
VALUES
    ('Lang',    'Apache Commons Lang',    'defects4j', 'java', 'String, numeric and reflection utilities.',        'utility,apache',    10),
    ('Math',    'Apache Commons Math',    'defects4j', 'java', 'Numeric analysis and mathematics helpers.',         'math,numeric',      20),
    ('Chart',   'JFreeChart',             'defects4j', 'java', 'Chart rendering library.',                          'chart,graphics',    30),
    ('Time',    'Joda-Time',              'defects4j', 'java', 'Date and time handling.',                           'time,datetime',     40),
    ('Cli',     'Apache Commons CLI',     'defects4j', 'java', 'Command-line argument parser.',                     'cli',               50),
    ('Closure', 'Google Closure Compiler','defects4j', 'java', 'JavaScript compiler and minifier.',                 'compiler',          60)
ON DUPLICATE KEY UPDATE
    `display_name` = VALUES(`display_name`),
    `language`     = VALUES(`language`),
    `description`  = VALUES(`description`),
    `tags`         = VALUES(`tags`),
    `sort_order`   = VALUES(`sort_order`);

-- Seed a small curated bug list per project (real Defects4J bug IDs).
INSERT INTO `benchmark_bugs` (`project_id`, `bug_key`, `title`, `severity`, `defects4j_project`, `defects4j_bug_id`, `description`, `tags`)
SELECT p.id, 'Lang-1',  'StringUtils regression (Lang-1)',  'normal', 'Lang', 1, 'Classical Defects4J reference bug from StringUtils.', 'stringutils' FROM `benchmark_projects` p WHERE p.project_code = 'Lang'
UNION ALL
SELECT p.id, 'Lang-10', 'NumberUtils corner case (Lang-10)','normal', 'Lang', 10,'Numeric parsing edge case.',                          'numberutils' FROM `benchmark_projects` p WHERE p.project_code = 'Lang'
UNION ALL
SELECT p.id, 'Math-2',  'StatUtils variance (Math-2)',      'normal', 'Math', 2, 'Variance computation with edge input.',               'stats'       FROM `benchmark_projects` p WHERE p.project_code = 'Math'
UNION ALL
SELECT p.id, 'Math-50', 'BigMatrix decomposition (Math-50)','hard',   'Math', 50,'Numeric stability issue in matrix decomposition.',    'matrix'      FROM `benchmark_projects` p WHERE p.project_code = 'Math'
UNION ALL
SELECT p.id, 'Chart-1', 'XYPlot renderer (Chart-1)',        'normal', 'Chart',1, 'Renderer bug in XYPlot.',                             'plot'        FROM `benchmark_projects` p WHERE p.project_code = 'Chart'
UNION ALL
SELECT p.id, 'Time-1',  'DateTime parse (Time-1)',          'normal', 'Time', 1, 'Parsing regression on leap seconds.',                 'parse'       FROM `benchmark_projects` p WHERE p.project_code = 'Time'
UNION ALL
SELECT p.id, 'Cli-5',   'Option parsing (Cli-5)',           'normal', 'Cli',  5, 'Short option parsing regression.',                    'parse'       FROM `benchmark_projects` p WHERE p.project_code = 'Cli'
UNION ALL
SELECT p.id, 'Closure-1','Type inference (Closure-1)',      'hard',   'Closure',1,'Type inference off-by-one bug.',                     'types'       FROM `benchmark_projects` p WHERE p.project_code = 'Closure'
ON DUPLICATE KEY UPDATE
    `title`       = VALUES(`title`),
    `severity`    = VALUES(`severity`),
    `description` = VALUES(`description`),
    `tags`        = VALUES(`tags`);

-- ---------------------------------------------------------------------------
-- Module: Personal Center & User Profile
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS `user_preferences` (
    `user_id` BIGINT UNSIGNED NOT NULL,
    `default_agent_model` VARCHAR(128) NULL,
    `default_chat_model` VARCHAR(128) NULL,
    `default_language` VARCHAR(24) NULL,
    `locale` VARCHAR(8) NOT NULL DEFAULT 'zh',
    `theme` VARCHAR(8) NOT NULL DEFAULT 'dark',
    `timezone` VARCHAR(64) NULL,
    `bio` VARCHAR(255) NULL,
    `show_site_map_widget` TINYINT(1) NOT NULL DEFAULT 1,
    `updated_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (`user_id`),
    CONSTRAINT `fk_user_preferences_user_id`
        FOREIGN KEY (`user_id`) REFERENCES `users` (`id`)
        ON DELETE CASCADE
        ON UPDATE CASCADE
);

CREATE TABLE IF NOT EXISTS `user_api_tokens` (
    `id` BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    `user_id` BIGINT UNSIGNED NOT NULL,
    `token_name` VARCHAR(64) NOT NULL,
    `token_prefix` VARCHAR(16) NOT NULL,
    `token_hash` VARCHAR(128) NOT NULL,
    `scope` VARCHAR(64) NOT NULL DEFAULT 'repair',
    `last_used_at` DATETIME NULL,
    `expires_at` DATETIME NULL,
    `revoked_at` DATETIME NULL,
    `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (`id`),
    UNIQUE KEY `uk_user_api_tokens_hash` (`token_hash`),
    KEY `idx_user_api_tokens_user_created` (`user_id`, `created_at`)
);

CREATE TABLE IF NOT EXISTS `user_pdf_exports` (
    `id` BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    `user_id` BIGINT UNSIGNED NOT NULL,
    `history_id` BIGINT UNSIGNED NULL,
    `benchmark_run_id` BIGINT UNSIGNED NULL,
    `export_type` VARCHAR(32) NOT NULL DEFAULT 'repair_report',
    `file_bytes` INT NOT NULL DEFAULT 0,
    `file_sha256` VARCHAR(64) NULL,
    `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (`id`),
    KEY `idx_user_pdf_exports_user_created` (`user_id`, `created_at`),
    CONSTRAINT `fk_user_pdf_exports_user_id`
        FOREIGN KEY (`user_id`) REFERENCES `users` (`id`)
        ON DELETE CASCADE
        ON UPDATE CASCADE
);
