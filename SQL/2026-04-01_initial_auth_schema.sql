CREATE DATABASE IF NOT EXISTS `AutoRepair`
  DEFAULT CHARACTER SET utf8mb4
  DEFAULT COLLATE utf8mb4_unicode_ci;

USE `AutoRepair`;

CREATE TABLE IF NOT EXISTS `users` (
    `id` BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    `email` VARCHAR(255) NOT NULL,
    `display_name` VARCHAR(120) NOT NULL,
    `password_hash` VARCHAR(255) NULL,
    `avatar_url` VARCHAR(512) NULL,
    `auth_source` VARCHAR(32) NOT NULL DEFAULT 'local',
    `role` VARCHAR(16) NOT NULL DEFAULT 'basic',
    `account_status` VARCHAR(16) NOT NULL DEFAULT 'active',
    `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    `updated_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    `last_login_at` DATETIME NULL DEFAULT NULL,
    PRIMARY KEY (`id`),
    UNIQUE KEY `uk_users_email` (`email`),
    KEY `idx_users_role_created` (`role`, `created_at`),
    KEY `idx_users_status_created` (`account_status`, `created_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS `oauth_accounts` (
    `id` BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    `user_id` BIGINT UNSIGNED NOT NULL,
    `provider` VARCHAR(32) NOT NULL,
    `provider_user_id` VARCHAR(191) NOT NULL,
    `email` VARCHAR(255) NULL,
    `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (`id`),
    UNIQUE KEY `uk_oauth_provider_user` (`provider`, `provider_user_id`),
    KEY `idx_oauth_accounts_user_id` (`user_id`),
    CONSTRAINT `fk_oauth_accounts_user_id`
        FOREIGN KEY (`user_id`) REFERENCES `users` (`id`)
        ON DELETE CASCADE
        ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS `conversation_histories` (
    `id` BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    `user_id` BIGINT UNSIGNED NOT NULL,
    `mode` VARCHAR(16) NOT NULL,
    `title` VARCHAR(255) NOT NULL,
    `preview_text` TEXT NULL,
    `model` VARCHAR(128) NULL,
    `language` VARCHAR(32) NULL,
    `snapshot_json` LONGTEXT NOT NULL,
    `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    `updated_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    `deleted_at` DATETIME NULL DEFAULT NULL,
    PRIMARY KEY (`id`),
    KEY `idx_conversation_histories_user_updated` (`user_id`, `updated_at`),
    KEY `idx_conversation_histories_user_deleted_updated` (`user_id`, `deleted_at`, `updated_at`),
    CONSTRAINT `fk_conversation_histories_user_id`
        FOREIGN KEY (`user_id`) REFERENCES `users` (`id`)
        ON DELETE CASCADE
        ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS `user_login_events` (
    `id` BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    `user_id` BIGINT UNSIGNED NULL,
    `email_attempt` VARCHAR(255) NULL,
    `login_method` VARCHAR(32) NOT NULL,
    `login_status` VARCHAR(16) NOT NULL DEFAULT 'success',
    `failure_reason` VARCHAR(255) NULL,
    `ip_address` VARCHAR(64) NULL,
    `user_agent` VARCHAR(255) NULL,
    `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (`id`),
    KEY `idx_user_login_events_created` (`created_at`),
    KEY `idx_user_login_events_user_created` (`user_id`, `created_at`),
    CONSTRAINT `fk_user_login_events_user_id`
        FOREIGN KEY (`user_id`) REFERENCES `users` (`id`)
        ON DELETE SET NULL
        ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS `llm_requests` (
    `id` BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    `user_id` BIGINT UNSIGNED NULL,
    `history_id` BIGINT UNSIGNED NULL,
    `request_mode` VARCHAR(24) NOT NULL,
    `stage` VARCHAR(32) NULL,
    `purpose` VARCHAR(64) NULL,
    `provider` VARCHAR(32) NOT NULL,
    `model` VARCHAR(128) NOT NULL,
    `source_type` VARCHAR(24) NULL,
    `is_streaming` TINYINT(1) NOT NULL DEFAULT 0,
    `is_json_response` TINYINT(1) NOT NULL DEFAULT 1,
    `request_status` VARCHAR(16) NOT NULL DEFAULT 'started',
    `token_source` VARCHAR(16) NULL,
    `prompt_chars` INT NOT NULL DEFAULT 0,
    `response_chars` INT NOT NULL DEFAULT 0,
    `latency_ms` INT NULL,
    `error_message` TEXT NULL,
    `started_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    `finished_at` DATETIME NULL DEFAULT NULL,
    PRIMARY KEY (`id`),
    KEY `idx_llm_requests_started` (`started_at`),
    KEY `idx_llm_requests_user_started` (`user_id`, `started_at`),
    KEY `idx_llm_requests_model_started` (`model`, `started_at`),
    KEY `idx_llm_requests_mode_started` (`request_mode`, `started_at`),
    KEY `idx_llm_requests_status_started` (`request_status`, `started_at`),
    CONSTRAINT `fk_llm_requests_user_id`
        FOREIGN KEY (`user_id`) REFERENCES `users` (`id`)
        ON DELETE SET NULL
        ON UPDATE CASCADE,
    CONSTRAINT `fk_llm_requests_history_id`
        FOREIGN KEY (`history_id`) REFERENCES `conversation_histories` (`id`)
        ON DELETE SET NULL
        ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS `llm_request_messages` (
    `id` BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    `request_id` BIGINT UNSIGNED NOT NULL,
    `system_prompt` LONGTEXT NULL,
    `prompt_text` LONGTEXT NOT NULL,
    `response_text` LONGTEXT NULL,
    `parsed_response_json` LONGTEXT NULL,
    `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (`id`),
    UNIQUE KEY `uk_llm_request_messages_request_id` (`request_id`),
    CONSTRAINT `fk_llm_request_messages_request_id`
        FOREIGN KEY (`request_id`) REFERENCES `llm_requests` (`id`)
        ON DELETE CASCADE
        ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS `llm_token_usage` (
    `id` BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    `request_id` BIGINT UNSIGNED NOT NULL,
    `input_tokens` INT NOT NULL DEFAULT 0,
    `output_tokens` INT NOT NULL DEFAULT 0,
    `total_tokens` INT NOT NULL DEFAULT 0,
    `cached_input_tokens` INT NOT NULL DEFAULT 0,
    `reasoning_tokens` INT NOT NULL DEFAULT 0,
    `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (`id`),
    UNIQUE KEY `uk_llm_token_usage_request_id` (`request_id`),
    KEY `idx_llm_token_usage_total_tokens` (`total_tokens`),
    CONSTRAINT `fk_llm_token_usage_request_id`
        FOREIGN KEY (`request_id`) REFERENCES `llm_requests` (`id`)
        ON DELETE CASCADE
        ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS `llm_request_tool_events` (
    `id` BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    `request_id` BIGINT UNSIGNED NOT NULL,
    `round_index` INT NULL,
    `status` VARCHAR(16) NOT NULL,
    `tool_name` VARCHAR(128) NOT NULL,
    `arguments_json` LONGTEXT NULL,
    `output_preview` LONGTEXT NULL,
    `output_truncated` TINYINT(1) NOT NULL DEFAULT 0,
    `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (`id`),
    KEY `idx_llm_request_tool_events_request_created` (`request_id`, `created_at`),
    CONSTRAINT `fk_llm_request_tool_events_request_id`
        FOREIGN KEY (`request_id`) REFERENCES `llm_requests` (`id`)
        ON DELETE CASCADE
        ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS `subscription_plans` (
    `id` BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    `plan_code` VARCHAR(64) NOT NULL,
    `plan_name` VARCHAR(120) NOT NULL,
    `role_granted` VARCHAR(16) NOT NULL DEFAULT 'advanced',
    `billing_cycle` VARCHAR(16) NOT NULL DEFAULT 'one_time',
    `amount_cents` INT NOT NULL,
    `currency` VARCHAR(8) NOT NULL DEFAULT 'CNY',
    `description` TEXT NULL,
    `is_active` TINYINT(1) NOT NULL DEFAULT 1,
    `sort_order` INT NOT NULL DEFAULT 0,
    `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    `updated_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (`id`),
    UNIQUE KEY `uk_subscription_plans_code` (`plan_code`),
    KEY `idx_subscription_plans_active_sort` (`is_active`, `sort_order`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS `payment_orders` (
    `id` BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    `order_no` VARCHAR(48) NOT NULL,
    `user_id` BIGINT UNSIGNED NOT NULL,
    `plan_id` BIGINT UNSIGNED NOT NULL,
    `plan_code` VARCHAR(64) NOT NULL,
    `plan_name_snapshot` VARCHAR(120) NOT NULL,
    `target_role` VARCHAR(16) NOT NULL DEFAULT 'advanced',
    `amount_cents` INT NOT NULL,
    `currency` VARCHAR(8) NOT NULL DEFAULT 'CNY',
    `payment_method` VARCHAR(16) NOT NULL,
    `order_status` VARCHAR(24) NOT NULL DEFAULT 'pending',
    `provider_status` VARCHAR(32) NULL,
    `checkout_action` VARCHAR(24) NOT NULL DEFAULT 'pending_provider_init',
    `checkout_url` VARCHAR(1024) NULL,
    `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    `updated_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    `paid_at` DATETIME NULL DEFAULT NULL,
    PRIMARY KEY (`id`),
    UNIQUE KEY `uk_payment_orders_order_no` (`order_no`),
    KEY `idx_payment_orders_user_created` (`user_id`, `created_at`),
    KEY `idx_payment_orders_status_paid` (`order_status`, `paid_at`),
    KEY `idx_payment_orders_method_created` (`payment_method`, `created_at`),
    CONSTRAINT `fk_payment_orders_user_id`
        FOREIGN KEY (`user_id`) REFERENCES `users` (`id`)
        ON DELETE CASCADE
        ON UPDATE CASCADE,
    CONSTRAINT `fk_payment_orders_plan_id`
        FOREIGN KEY (`plan_id`) REFERENCES `subscription_plans` (`id`)
        ON DELETE RESTRICT
        ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS `payment_provider_sessions` (
    `id` BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    `order_id` BIGINT UNSIGNED NOT NULL,
    `provider` VARCHAR(16) NOT NULL,
    `session_status` VARCHAR(24) NOT NULL DEFAULT 'ready',
    `provider_session_id` VARCHAR(128) NULL,
    `provider_reference` VARCHAR(128) NULL,
    `redirect_url` VARCHAR(1024) NULL,
    `qr_code_text` TEXT NULL,
    `session_payload_json` LONGTEXT NULL,
    `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    `updated_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (`id`),
    KEY `idx_payment_provider_sessions_order_created` (`order_id`, `created_at`),
    CONSTRAINT `fk_payment_provider_sessions_order_id`
        FOREIGN KEY (`order_id`) REFERENCES `payment_orders` (`id`)
        ON DELETE CASCADE
        ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS `payment_transactions` (
    `id` BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    `order_id` BIGINT UNSIGNED NOT NULL,
    `transaction_no` VARCHAR(64) NOT NULL,
    `provider` VARCHAR(16) NOT NULL,
    `payment_method` VARCHAR(16) NOT NULL,
    `transaction_status` VARCHAR(24) NOT NULL DEFAULT 'paid',
    `amount_cents` INT NOT NULL,
    `currency` VARCHAR(8) NOT NULL DEFAULT 'CNY',
    `provider_reference` VARCHAR(128) NULL,
    `raw_payload_json` LONGTEXT NULL,
    `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    `paid_at` DATETIME NULL DEFAULT NULL,
    PRIMARY KEY (`id`),
    UNIQUE KEY `uk_payment_transactions_transaction_no` (`transaction_no`),
    KEY `idx_payment_transactions_order_created` (`order_id`, `created_at`),
    CONSTRAINT `fk_payment_transactions_order_id`
        FOREIGN KEY (`order_id`) REFERENCES `payment_orders` (`id`)
        ON DELETE CASCADE
        ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS `user_subscriptions` (
    `id` BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    `user_id` BIGINT UNSIGNED NOT NULL,
    `plan_id` BIGINT UNSIGNED NOT NULL,
    `plan_code` VARCHAR(64) NOT NULL,
    `role_granted` VARCHAR(16) NOT NULL DEFAULT 'advanced',
    `subscription_status` VARCHAR(24) NOT NULL DEFAULT 'active',
    `starts_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    `ends_at` DATETIME NULL DEFAULT NULL,
    `revoked_at` DATETIME NULL DEFAULT NULL,
    `activated_by_order_id` BIGINT UNSIGNED NULL,
    `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    `updated_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (`id`),
    KEY `idx_user_subscriptions_user_status` (`user_id`, `subscription_status`, `starts_at`),
    KEY `idx_user_subscriptions_plan_status` (`plan_id`, `subscription_status`, `starts_at`),
    CONSTRAINT `fk_user_subscriptions_user_id`
        FOREIGN KEY (`user_id`) REFERENCES `users` (`id`)
        ON DELETE CASCADE
        ON UPDATE CASCADE,
    CONSTRAINT `fk_user_subscriptions_plan_id`
        FOREIGN KEY (`plan_id`) REFERENCES `subscription_plans` (`id`)
        ON DELETE RESTRICT
        ON UPDATE CASCADE,
    CONSTRAINT `fk_user_subscriptions_activated_order_id`
        FOREIGN KEY (`activated_by_order_id`) REFERENCES `payment_orders` (`id`)
        ON DELETE SET NULL
        ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS `payment_events` (
    `id` BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    `order_id` BIGINT UNSIGNED NOT NULL,
    `event_type` VARCHAR(64) NOT NULL,
    `actor_user_id` BIGINT UNSIGNED NULL,
    `actor_role` VARCHAR(24) NOT NULL DEFAULT 'system',
    `event_payload_json` LONGTEXT NULL,
    `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (`id`),
    KEY `idx_payment_events_order_created` (`order_id`, `created_at`),
    KEY `idx_payment_events_actor_created` (`actor_user_id`, `created_at`),
    CONSTRAINT `fk_payment_events_order_id`
        FOREIGN KEY (`order_id`) REFERENCES `payment_orders` (`id`)
        ON DELETE CASCADE
        ON UPDATE CASCADE,
    CONSTRAINT `fk_payment_events_actor_user_id`
        FOREIGN KEY (`actor_user_id`) REFERENCES `users` (`id`)
        ON DELETE SET NULL
        ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS `user_role_grants` (
    `id` BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    `user_id` BIGINT UNSIGNED NOT NULL,
    `granted_by_user_id` BIGINT UNSIGNED NULL,
    `previous_role` VARCHAR(16) NOT NULL,
    `new_role` VARCHAR(16) NOT NULL,
    `grant_source` VARCHAR(32) NOT NULL,
    `payment_order_id` BIGINT UNSIGNED NULL,
    `note` VARCHAR(255) NULL,
    `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (`id`),
    KEY `idx_user_role_grants_user_created` (`user_id`, `created_at`),
    KEY `idx_user_role_grants_granter_created` (`granted_by_user_id`, `created_at`),
    CONSTRAINT `fk_user_role_grants_user_id`
        FOREIGN KEY (`user_id`) REFERENCES `users` (`id`)
        ON DELETE CASCADE
        ON UPDATE CASCADE,
    CONSTRAINT `fk_user_role_grants_granted_by_user_id`
        FOREIGN KEY (`granted_by_user_id`) REFERENCES `users` (`id`)
        ON DELETE SET NULL
        ON UPDATE CASCADE,
    CONSTRAINT `fk_user_role_grants_payment_order_id`
        FOREIGN KEY (`payment_order_id`) REFERENCES `payment_orders` (`id`)
        ON DELETE SET NULL
        ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

INSERT INTO `subscription_plans` (
    `plan_code`,
    `plan_name`,
    `role_granted`,
    `billing_cycle`,
    `amount_cents`,
    `currency`,
    `description`,
    `is_active`,
    `sort_order`
)
VALUES (
    'advanced_access',
    'Advanced Access',
    'advanced',
    'one_time',
    19900,
    'CNY',
    'Unlock the advanced tier with priority usage and access to premium workflows.',
    1,
    10
)
ON DUPLICATE KEY UPDATE
    `plan_name` = VALUES(`plan_name`),
    `role_granted` = VALUES(`role_granted`),
    `billing_cycle` = VALUES(`billing_cycle`),
    `amount_cents` = VALUES(`amount_cents`),
    `currency` = VALUES(`currency`),
    `description` = VALUES(`description`),
    `is_active` = VALUES(`is_active`),
    `sort_order` = VALUES(`sort_order`);
