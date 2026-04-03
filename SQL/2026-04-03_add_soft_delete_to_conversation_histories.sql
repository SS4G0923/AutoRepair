USE `AutoRepair`;

ALTER TABLE `conversation_histories`
    ADD COLUMN `deleted_at` DATETIME NULL DEFAULT NULL AFTER `updated_at`;

ALTER TABLE `conversation_histories`
    ADD KEY `idx_conversation_histories_user_deleted_updated` (`user_id`, `deleted_at`, `updated_at`);
