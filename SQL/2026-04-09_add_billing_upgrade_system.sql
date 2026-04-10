USE `AutoRepair`;

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
);

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
);

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
);

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
);

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
);

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
);

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
);

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
