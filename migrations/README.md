# Migration Scripts

This directory contains database migration scripts for the loan management system.

## Available Migrations

### add_sync_fields.py
Adds synchronization-related fields to the User and Loan tables to support API integration.

**Fields Added:**
- `server_id`: Unique identifier from the remote server
- `sync_status`: Current synchronization status (pending, synced, failed)
- `last_synced_at`: Timestamp of last successful sync
- `version`: Version number for conflict resolution
- `is_deleted`: Soft delete flag
- `sync_retry_count`: Number of sync retry attempts
- `sync_error_message`: Error message from last failed sync

**Usage:**
```bash
# Run migration
python migrations/add_sync_fields.py

# Rollback migration
python migrations/add_sync_fields.py --rollback
```

## Running Migrations

1. Ensure your database connection is configured in `config/settings.py`
2. Make sure all dependencies are installed (`pip install -r requirements.txt`)
3. Run the migration script from the project root directory

## Migration Best Practices

1. Always backup your database before running migrations
2. Test migrations on a development environment first
3. Review the SQL statements in the migration files before execution
4. Keep migration scripts in version control
5. Document any manual steps required after migration

## Troubleshooting

If a migration fails:
1. Check the error message for specific details
2. Verify database connectivity and permissions
3. Ensure the database schema is in the expected state
4. Use the rollback option if available to revert changes
5. Contact the development team if issues persist