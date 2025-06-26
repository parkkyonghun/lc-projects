"""Initial database schema with users, customers, loan officers, loans, and payments

Revision ID: bb65b9a9c763
Revises: 
Create Date: 2025-06-26 15:55:08.967346

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'bb65b9a9c763'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Drop all existing tables in the correct order to avoid foreign key violations
    # First drop all the old tables from the previous schema
    op.execute('DROP TABLE IF EXISTS "UserBranchAssignment" CASCADE')
    op.execute('DROP TABLE IF EXISTS "UserRole" CASCADE')
    op.execute('DROP TABLE IF EXISTS "Role" CASCADE')
    op.execute('DROP TABLE IF EXISTS "NotificationEvent" CASCADE')
    op.execute('DROP TABLE IF EXISTS "UserActivity" CASCADE')
    op.execute('DROP TABLE IF EXISTS "Report" CASCADE')
    op.execute('DROP TABLE IF EXISTS "ReportComment" CASCADE')
    op.execute('DROP TABLE IF EXISTS "InAppNotification" CASCADE')
    op.execute('DROP TABLE IF EXISTS "PushSubscription" CASCADE')
    op.execute('DROP TABLE IF EXISTS "TelegramLinkingCode" CASCADE')
    op.execute('DROP TABLE IF EXISTS "TelegramSubscription" CASCADE')
    op.execute('DROP TABLE IF EXISTS "ActivityLog" CASCADE')
    op.execute('DROP TABLE IF EXISTS "OrganizationSettings" CASCADE')
    op.execute('DROP TABLE IF EXISTS "Payment" CASCADE')
    op.execute('DROP TABLE IF EXISTS "Loan" CASCADE')
    op.execute('DROP TABLE IF EXISTS "User" CASCADE')
    op.execute('DROP TABLE IF EXISTS "Branch" CASCADE')
    op.execute('DROP TABLE IF EXISTS "_prisma_migrations" CASCADE')
    
    # Now create the new tables
    op.create_table('branches',
    sa.Column('id', sa.String(), nullable=False),
    sa.Column('code', sa.String(length=20), nullable=False),
    sa.Column('name', sa.String(length=100), nullable=False),
    sa.Column('description', sa.Text(), nullable=True),
    sa.Column('address', sa.String(length=255), nullable=True),
    sa.Column('city', sa.String(length=100), nullable=True),
    sa.Column('state', sa.String(length=100), nullable=True),
    sa.Column('country', sa.String(length=100), nullable=True),
    sa.Column('postal_code', sa.String(length=20), nullable=True),
    sa.Column('phone', sa.String(length=20), nullable=True),
    sa.Column('email', sa.String(length=100), nullable=True),
    sa.Column('parent_id', sa.String(), nullable=True),
    sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=True),
    sa.Column('updated_at', sa.DateTime(), server_default=sa.text('now()'), nullable=True),
    sa.Column('is_active', sa.Boolean(), nullable=True),
    sa.ForeignKeyConstraint(['parent_id'], ['branches.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_branches_code'), 'branches', ['code'], unique=True)
    op.create_table('users',
    sa.Column('id', sa.String(), nullable=False),
    sa.Column('user_type', sa.Enum('CUSTOMER', 'LOAN_OFFICER', 'ADMIN', name='usertype'), nullable=False),
    sa.Column('username', sa.String(length=50), nullable=False),
    sa.Column('email', sa.String(length=100), nullable=False),
    sa.Column('phone_number', sa.String(length=20), nullable=False),
    sa.Column('hashed_password', sa.String(length=255), nullable=False),
    sa.Column('branch_id', sa.String(), nullable=True),
    sa.Column('last_login', sa.DateTime(), nullable=True),
    sa.Column('is_active', sa.Boolean(), nullable=False),
    sa.Column('is_verified', sa.Boolean(), nullable=False),
    sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=True),
    sa.Column('updated_at', sa.DateTime(), server_default=sa.text('now()'), nullable=True),
    sa.ForeignKeyConstraint(['branch_id'], ['branches.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('email'),
    sa.UniqueConstraint('phone_number')
    )
    op.create_index(op.f('ix_users_branch_id'), 'users', ['branch_id'], unique=False)
    op.create_index(op.f('ix_users_username'), 'users', ['username'], unique=True)
    op.create_table('customers',
    sa.Column('id', sa.String(), nullable=False),
    sa.Column('khmer_name', sa.String(), nullable=False),
    sa.Column('english_name', sa.String(), nullable=False),
    sa.Column('id_card_number', sa.String(), nullable=False),
    sa.Column('address', sa.String(), nullable=True),
    sa.Column('occupation', sa.String(), nullable=True),
    sa.Column('monthly_income', sa.Float(), nullable=True),
    sa.Column('id_card_photo_url', sa.String(), nullable=True),
    sa.Column('profile_photo_url', sa.String(), nullable=True),
    sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=True),
    sa.Column('updated_at', sa.DateTime(), server_default=sa.text('now()'), nullable=True),
    sa.Column('is_active', sa.Boolean(), nullable=True),
    sa.ForeignKeyConstraint(['id'], ['users.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('id_card_number')
    )
    op.create_table('loan_officers',
    sa.Column('id', sa.String(), nullable=False),
    sa.Column('employee_id', sa.String(), nullable=False),
    sa.Column('hire_date', sa.DateTime(), nullable=False),
    sa.Column('department', sa.String(), nullable=True),
    sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=True),
    sa.Column('updated_at', sa.DateTime(), server_default=sa.text('now()'), nullable=True),
    sa.Column('is_active', sa.Boolean(), nullable=True),
    sa.ForeignKeyConstraint(['id'], ['users.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('employee_id')
    )
    op.create_table('loans',
    sa.Column('id', sa.String(), nullable=False),
    sa.Column('customer_id', sa.String(), nullable=False),
    sa.Column('loan_officer_id', sa.String(), nullable=True),
    sa.Column('branch_id', sa.String(), nullable=False),
    sa.Column('loan_amount', sa.Float(), nullable=False),
    sa.Column('interest_rate', sa.Float(), nullable=False),
    sa.Column('term_months', sa.Integer(), nullable=False),
    sa.Column('monthly_payment', sa.Float(), nullable=True),
    sa.Column('purpose', sa.String(), nullable=True),
    sa.Column('status', sa.Enum('PENDING', 'APPROVED', 'ACTIVE', 'COMPLETED', 'REJECTED', 'DEFAULTED', name='loanstatus'), nullable=False),
    sa.Column('application_date', sa.DateTime(), server_default=sa.text('now()'), nullable=True),
    sa.Column('approval_date', sa.DateTime(), nullable=True),
    sa.Column('start_date', sa.DateTime(), nullable=True),
    sa.Column('next_payment_date', sa.DateTime(), nullable=True),
    sa.Column('end_date', sa.DateTime(), nullable=True),
    sa.Column('remaining_balance', sa.Float(), nullable=True),
    sa.Column('total_paid', sa.Float(), nullable=True),
    sa.Column('collateral_description', sa.String(), nullable=True),
    sa.Column('collateral_value', sa.Float(), nullable=True),
    sa.Column('repayment_schedule', sa.JSON(), nullable=True),
    sa.Column('notes', sa.String(), nullable=True),
    sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=True),
    sa.Column('updated_at', sa.DateTime(), server_default=sa.text('now()'), nullable=True),
    sa.Column('is_active', sa.Boolean(), nullable=True),
    sa.ForeignKeyConstraint(['branch_id'], ['branches.id'], ),
    sa.ForeignKeyConstraint(['customer_id'], ['customers.id'], ),
    sa.ForeignKeyConstraint(['loan_officer_id'], ['loan_officers.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_loans_branch_id'), 'loans', ['branch_id'], unique=False)
    op.create_index(op.f('ix_loans_customer_id'), 'loans', ['customer_id'], unique=False)
    op.create_index(op.f('ix_loans_loan_officer_id'), 'loans', ['loan_officer_id'], unique=False)
    op.create_table('payments',
    sa.Column('id', sa.String(), nullable=False),
    sa.Column('loan_id', sa.String(), nullable=False),
    sa.Column('collected_by', sa.String(), nullable=True),
    sa.Column('amount', sa.Numeric(precision=12, scale=2), nullable=False),
    sa.Column('payment_date', sa.DateTime(), nullable=False),
    sa.Column('payment_method', postgresql.ENUM('CASH', 'BANK_TRANSFER', 'MOBILE_PAYMENT', 'CHECK', 'CREDIT_CARD', 'WALLET', name='payment_method_enum'), nullable=False),
    sa.Column('status', postgresql.ENUM('PENDING', 'COMPLETED', 'FAILED', 'REFUNDED', 'PARTIALLY_REFUNDED', 'CANCELLED', name='payment_status_enum'), nullable=False),
    sa.Column('reference_number', sa.String(length=50), nullable=True),
    sa.Column('receipt_number', sa.String(length=50), nullable=True),
    sa.Column('transaction_id', sa.String(length=100), nullable=True),
    sa.Column('principal_amount', sa.Numeric(precision=12, scale=2), nullable=False),
    sa.Column('interest_amount', sa.Numeric(precision=12, scale=2), nullable=False),
    sa.Column('late_fee', sa.Numeric(precision=12, scale=2), nullable=True),
    sa.Column('other_fees', sa.Numeric(precision=12, scale=2), nullable=True),
    sa.Column('notes', sa.Text(), nullable=True),
    sa.Column('attachment_url', sa.String(length=255), nullable=True),
    sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=True),
    sa.Column('updated_at', sa.DateTime(), server_default=sa.text('now()'), nullable=True),
    sa.Column('is_active', sa.Boolean(), nullable=True),
    sa.ForeignKeyConstraint(['collected_by'], ['users.id'], ),
    sa.ForeignKeyConstraint(['loan_id'], ['loans.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('receipt_number'),
    sa.UniqueConstraint('reference_number')
    )
    op.create_index(op.f('ix_payments_collected_by'), 'payments', ['collected_by'], unique=False)
    op.create_index(op.f('ix_payments_loan_id'), 'payments', ['loan_id'], unique=False)
    # ### end Alembic commands ###


def downgrade() -> None:
    """Downgrade schema."""
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('TelegramSubscription',
    sa.Column('id', sa.TEXT(), autoincrement=False, nullable=False),
    sa.Column('userId', sa.TEXT(), autoincrement=False, nullable=False),
    sa.Column('chatId', sa.TEXT(), autoincrement=False, nullable=False),
    sa.Column('username', sa.TEXT(), autoincrement=False, nullable=True),
    sa.Column('createdAt', postgresql.TIMESTAMP(precision=3), server_default=sa.text('CURRENT_TIMESTAMP'), autoincrement=False, nullable=False),
    sa.Column('updatedAt', postgresql.TIMESTAMP(precision=3), autoincrement=False, nullable=False),
    sa.ForeignKeyConstraint(['userId'], ['User.id'], name=op.f('TelegramSubscription_userId_fkey'), onupdate='CASCADE', ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('TelegramSubscription_pkey'))
    )
    op.create_index(op.f('TelegramSubscription_userId_key'), 'TelegramSubscription', ['userId'], unique=True)
    op.create_index(op.f('TelegramSubscription_userId_idx'), 'TelegramSubscription', ['userId'], unique=False)
    op.create_index(op.f('TelegramSubscription_chatId_key'), 'TelegramSubscription', ['chatId'], unique=True)
    op.create_table('ReportComment',
    sa.Column('id', sa.TEXT(), autoincrement=False, nullable=False),
    sa.Column('reportId', sa.TEXT(), autoincrement=False, nullable=False),
    sa.Column('userId', sa.TEXT(), autoincrement=False, nullable=False),
    sa.Column('content', sa.TEXT(), autoincrement=False, nullable=False),
    sa.Column('createdAt', postgresql.TIMESTAMP(precision=3), server_default=sa.text('CURRENT_TIMESTAMP'), autoincrement=False, nullable=False),
    sa.Column('updatedAt', postgresql.TIMESTAMP(precision=3), autoincrement=False, nullable=False),
    sa.Column('parentId', sa.TEXT(), autoincrement=False, nullable=True),
    sa.ForeignKeyConstraint(['parentId'], ['ReportComment.id'], name=op.f('ReportComment_parentId_fkey'), onupdate='CASCADE', ondelete='SET NULL'),
    sa.ForeignKeyConstraint(['reportId'], ['Report.id'], name=op.f('ReportComment_reportId_fkey'), onupdate='CASCADE', ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['userId'], ['User.id'], name=op.f('ReportComment_userId_fkey'), onupdate='CASCADE', ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('ReportComment_pkey'))
    )
    op.create_index(op.f('ReportComment_userId_idx'), 'ReportComment', ['userId'], unique=False)
    op.create_index(op.f('ReportComment_reportId_idx'), 'ReportComment', ['reportId'], unique=False)
    op.create_index(op.f('ReportComment_parentId_idx'), 'ReportComment', ['parentId'], unique=False)
    op.create_table('ActivityLog',
    sa.Column('id', sa.TEXT(), autoincrement=False, nullable=False),
    sa.Column('userId', sa.TEXT(), autoincrement=False, nullable=False),
    sa.Column('action', sa.TEXT(), autoincrement=False, nullable=False),
    sa.Column('details', sa.TEXT(), autoincrement=False, nullable=True),
    sa.Column('ipAddress', sa.TEXT(), autoincrement=False, nullable=True),
    sa.Column('userAgent', sa.TEXT(), autoincrement=False, nullable=True),
    sa.Column('timestamp', postgresql.TIMESTAMP(precision=3), server_default=sa.text('CURRENT_TIMESTAMP'), autoincrement=False, nullable=False),
    sa.ForeignKeyConstraint(['userId'], ['User.id'], name=op.f('ActivityLog_userId_fkey'), onupdate='CASCADE', ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('ActivityLog_pkey'))
    )
    op.create_table('InAppNotification',
    sa.Column('id', sa.TEXT(), autoincrement=False, nullable=False),
    sa.Column('userId', sa.TEXT(), autoincrement=False, nullable=False),
    sa.Column('title', sa.TEXT(), autoincrement=False, nullable=False),
    sa.Column('body', sa.TEXT(), autoincrement=False, nullable=False),
    sa.Column('type', sa.TEXT(), autoincrement=False, nullable=False),
    sa.Column('data', postgresql.JSONB(astext_type=sa.Text()), autoincrement=False, nullable=True),
    sa.Column('isRead', sa.BOOLEAN(), server_default=sa.text('false'), autoincrement=False, nullable=False),
    sa.Column('createdAt', postgresql.TIMESTAMP(precision=3), server_default=sa.text('CURRENT_TIMESTAMP'), autoincrement=False, nullable=False),
    sa.Column('readAt', postgresql.TIMESTAMP(precision=3), autoincrement=False, nullable=True),
    sa.Column('actionUrl', sa.TEXT(), autoincrement=False, nullable=True),
    sa.ForeignKeyConstraint(['userId'], ['User.id'], name='InAppNotification_userId_fkey', onupdate='CASCADE', ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name='InAppNotification_pkey'),
    postgresql_ignore_search_path=False
    )
    op.create_index(op.f('InAppNotification_userId_idx'), 'InAppNotification', ['userId'], unique=False)
    op.create_index(op.f('InAppNotification_type_idx'), 'InAppNotification', ['type'], unique=False)
    op.create_index(op.f('InAppNotification_isRead_idx'), 'InAppNotification', ['isRead'], unique=False)
    op.create_index(op.f('InAppNotification_createdAt_idx'), 'InAppNotification', ['createdAt'], unique=False)
    op.create_table('Payment',
    sa.Column('id', sa.VARCHAR(), autoincrement=False, nullable=False),
    sa.Column('loan_id', sa.VARCHAR(), autoincrement=False, nullable=False),
    sa.Column('amount', sa.DOUBLE_PRECISION(precision=53), autoincrement=False, nullable=False),
    sa.Column('payment_date', postgresql.TIMESTAMP(), autoincrement=False, nullable=False),
    sa.Column('payment_method', postgresql.ENUM('cash', 'bank_transfer', 'mobile_payment', 'check', name='paymentmethod'), autoincrement=False, nullable=False),
    sa.Column('receipt_number', sa.VARCHAR(), autoincrement=False, nullable=True),
    sa.Column('notes', sa.VARCHAR(), autoincrement=False, nullable=True),
    sa.Column('late_fee', sa.DOUBLE_PRECISION(precision=53), autoincrement=False, nullable=True),
    sa.Column('createdAt', postgresql.TIMESTAMP(), server_default=sa.text('now()'), autoincrement=False, nullable=True),
    sa.Column('updatedAt', postgresql.TIMESTAMP(), server_default=sa.text('now()'), autoincrement=False, nullable=True),
    sa.ForeignKeyConstraint(['loan_id'], ['Loan.id'], name=op.f('Payment_loan_id_fkey')),
    sa.PrimaryKeyConstraint('id', name=op.f('Payment_pkey'))
    )
    op.create_table('Branch',
    sa.Column('id', sa.TEXT(), autoincrement=False, nullable=False),
    sa.Column('code', sa.TEXT(), autoincrement=False, nullable=False),
    sa.Column('name', sa.TEXT(), autoincrement=False, nullable=False),
    sa.Column('isActive', sa.BOOLEAN(), server_default=sa.text('true'), autoincrement=False, nullable=False),
    sa.Column('createdAt', postgresql.TIMESTAMP(precision=3), server_default=sa.text('CURRENT_TIMESTAMP'), autoincrement=False, nullable=False),
    sa.Column('updatedAt', postgresql.TIMESTAMP(precision=3), autoincrement=False, nullable=False),
    sa.Column('parentId', sa.TEXT(), autoincrement=False, nullable=True),
    sa.ForeignKeyConstraint(['parentId'], ['Branch.id'], name='Branch_parentId_fkey', onupdate='CASCADE', ondelete='SET NULL'),
    sa.PrimaryKeyConstraint('id', name='Branch_pkey'),
    postgresql_ignore_search_path=False
    )
    op.create_index(op.f('Branch_code_key'), 'Branch', ['code'], unique=True)
    op.create_table('TelegramLinkingCode',
    sa.Column('id', sa.TEXT(), autoincrement=False, nullable=False),
    sa.Column('code', sa.TEXT(), autoincrement=False, nullable=False),
    sa.Column('userId', sa.TEXT(), autoincrement=False, nullable=False),
    sa.Column('expiresAt', postgresql.TIMESTAMP(precision=3), autoincrement=False, nullable=False),
    sa.Column('createdAt', postgresql.TIMESTAMP(precision=3), server_default=sa.text('CURRENT_TIMESTAMP'), autoincrement=False, nullable=False),
    sa.ForeignKeyConstraint(['userId'], ['User.id'], name=op.f('TelegramLinkingCode_userId_fkey'), onupdate='CASCADE', ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('TelegramLinkingCode_pkey'))
    )
    op.create_index(op.f('TelegramLinkingCode_userId_idx'), 'TelegramLinkingCode', ['userId'], unique=False)
    op.create_index(op.f('TelegramLinkingCode_expiresAt_idx'), 'TelegramLinkingCode', ['expiresAt'], unique=False)
    op.create_index(op.f('TelegramLinkingCode_code_key'), 'TelegramLinkingCode', ['code'], unique=True)
    op.create_table('OrganizationSettings',
    sa.Column('id', sa.TEXT(), autoincrement=False, nullable=False),
    sa.Column('organizationId', sa.TEXT(), autoincrement=False, nullable=False),
    sa.Column('validationRules', postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text('\'{"comments": {"required": true, "minLength": 10}, "writeOffs": {"maxAmount": 1000, "requireApproval": true}, "ninetyPlus": {"maxAmount": 5000, "requireApproval": true}, "duplicateCheck": {"enabled": true}}\'::jsonb'), autoincrement=False, nullable=False),
    sa.Column('createdAt', postgresql.TIMESTAMP(precision=3), server_default=sa.text('CURRENT_TIMESTAMP'), autoincrement=False, nullable=False),
    sa.Column('updatedAt', postgresql.TIMESTAMP(precision=3), autoincrement=False, nullable=False),
    sa.PrimaryKeyConstraint('id', name=op.f('OrganizationSettings_pkey'))
    )
    op.create_index(op.f('OrganizationSettings_organizationId_key'), 'OrganizationSettings', ['organizationId'], unique=True)
    op.create_table('UserRole',
    sa.Column('id', sa.TEXT(), autoincrement=False, nullable=False),
    sa.Column('userId', sa.TEXT(), autoincrement=False, nullable=False),
    sa.Column('roleId', sa.TEXT(), autoincrement=False, nullable=False),
    sa.Column('branchId', sa.TEXT(), autoincrement=False, nullable=True),
    sa.Column('isDefault', sa.BOOLEAN(), server_default=sa.text('false'), autoincrement=False, nullable=False),
    sa.Column('createdAt', postgresql.TIMESTAMP(precision=3), server_default=sa.text('CURRENT_TIMESTAMP'), autoincrement=False, nullable=False),
    sa.Column('updatedAt', postgresql.TIMESTAMP(precision=3), autoincrement=False, nullable=False),
    sa.ForeignKeyConstraint(['branchId'], ['Branch.id'], name=op.f('UserRole_branchId_fkey'), onupdate='CASCADE', ondelete='SET NULL'),
    sa.ForeignKeyConstraint(['roleId'], ['Role.id'], name=op.f('UserRole_roleId_fkey'), onupdate='CASCADE', ondelete='RESTRICT'),
    sa.ForeignKeyConstraint(['userId'], ['User.id'], name=op.f('UserRole_userId_fkey'), onupdate='CASCADE', ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('UserRole_pkey'))
    )
    op.create_index(op.f('UserRole_userId_roleId_branchId_key'), 'UserRole', ['userId', 'roleId', 'branchId'], unique=True)
    op.create_index(op.f('UserRole_userId_idx'), 'UserRole', ['userId'], unique=False)
    op.create_table('Report',
    sa.Column('id', sa.TEXT(), autoincrement=False, nullable=False),
    sa.Column('branchId', sa.TEXT(), autoincrement=False, nullable=False),
    sa.Column('writeOffs', sa.DOUBLE_PRECISION(precision=53), autoincrement=False, nullable=False),
    sa.Column('ninetyPlus', sa.DOUBLE_PRECISION(precision=53), autoincrement=False, nullable=False),
    sa.Column('reportType', sa.TEXT(), server_default=sa.text("'actual'::text"), autoincrement=False, nullable=False),
    sa.Column('status', sa.TEXT(), server_default=sa.text("'pending'::text"), autoincrement=False, nullable=False),
    sa.Column('submittedBy', sa.TEXT(), autoincrement=False, nullable=False),
    sa.Column('comments', sa.TEXT(), autoincrement=False, nullable=True),
    sa.Column('createdAt', postgresql.TIMESTAMP(precision=3), server_default=sa.text('CURRENT_TIMESTAMP'), autoincrement=False, nullable=False),
    sa.Column('updatedAt', postgresql.TIMESTAMP(precision=3), autoincrement=False, nullable=False),
    sa.Column('planReportId', sa.TEXT(), autoincrement=False, nullable=True),
    sa.Column('submittedAt', sa.TEXT(), autoincrement=False, nullable=False),
    sa.Column('date', sa.DATE(), autoincrement=False, nullable=False),
    sa.ForeignKeyConstraint(['branchId'], ['Branch.id'], name=op.f('Report_branchId_fkey'), onupdate='CASCADE', ondelete='RESTRICT'),
    sa.ForeignKeyConstraint(['planReportId'], ['Report.id'], name=op.f('Report_planReportId_fkey'), onupdate='CASCADE', ondelete='SET NULL'),
    sa.PrimaryKeyConstraint('id', name=op.f('Report_pkey'))
    )
    op.create_index(op.f('Report_planReportId_idx'), 'Report', ['planReportId'], unique=False)
    op.create_index(op.f('Report_date_status_idx'), 'Report', ['date', 'status'], unique=False)
    op.create_index(op.f('Report_date_branchId_reportType_key'), 'Report', ['date', 'branchId', 'reportType'], unique=True)
    op.create_index(op.f('Report_branchId_date_idx'), 'Report', ['branchId', 'date'], unique=False)
    op.create_table('UserActivity',
    sa.Column('id', sa.TEXT(), autoincrement=False, nullable=False),
    sa.Column('userId', sa.TEXT(), autoincrement=False, nullable=False),
    sa.Column('action', sa.TEXT(), autoincrement=False, nullable=False),
    sa.Column('details', postgresql.JSONB(astext_type=sa.Text()), autoincrement=False, nullable=False),
    sa.Column('ipAddress', sa.TEXT(), autoincrement=False, nullable=False),
    sa.Column('userAgent', sa.TEXT(), autoincrement=False, nullable=False),
    sa.Column('createdAt', postgresql.TIMESTAMP(precision=3), server_default=sa.text('CURRENT_TIMESTAMP'), autoincrement=False, nullable=False),
    sa.ForeignKeyConstraint(['userId'], ['User.id'], name=op.f('UserActivity_userId_fkey'), onupdate='CASCADE', ondelete='RESTRICT'),
    sa.PrimaryKeyConstraint('id', name=op.f('UserActivity_pkey'))
    )
    op.create_index(op.f('UserActivity_userId_idx'), 'UserActivity', ['userId'], unique=False)
    op.create_index(op.f('UserActivity_createdAt_idx'), 'UserActivity', ['createdAt'], unique=False)
    op.create_index(op.f('UserActivity_action_idx'), 'UserActivity', ['action'], unique=False)
    op.create_table('NotificationEvent',
    sa.Column('id', sa.TEXT(), autoincrement=False, nullable=False),
    sa.Column('notificationId', sa.TEXT(), autoincrement=False, nullable=False),
    sa.Column('event', sa.TEXT(), autoincrement=False, nullable=False),
    sa.Column('metadata', postgresql.JSONB(astext_type=sa.Text()), autoincrement=False, nullable=True),
    sa.Column('timestamp', postgresql.TIMESTAMP(precision=3), server_default=sa.text('CURRENT_TIMESTAMP'), autoincrement=False, nullable=False),
    sa.ForeignKeyConstraint(['notificationId'], ['InAppNotification.id'], name=op.f('NotificationEvent_notificationId_fkey'), onupdate='CASCADE', ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('NotificationEvent_pkey'))
    )
    op.create_index(op.f('NotificationEvent_timestamp_idx'), 'NotificationEvent', ['timestamp'], unique=False)
    op.create_index(op.f('NotificationEvent_notificationId_idx'), 'NotificationEvent', ['notificationId'], unique=False)
    op.create_index(op.f('NotificationEvent_event_idx'), 'NotificationEvent', ['event'], unique=False)
    op.create_table('PushSubscription',
    sa.Column('id', sa.TEXT(), autoincrement=False, nullable=False),
    sa.Column('endpoint', sa.TEXT(), autoincrement=False, nullable=False),
    sa.Column('p256dh', sa.TEXT(), autoincrement=False, nullable=False),
    sa.Column('auth', sa.TEXT(), autoincrement=False, nullable=False),
    sa.Column('createdAt', postgresql.TIMESTAMP(precision=3), server_default=sa.text('CURRENT_TIMESTAMP'), autoincrement=False, nullable=False),
    sa.Column('updatedAt', postgresql.TIMESTAMP(precision=3), autoincrement=False, nullable=False),
    sa.Column('userId', sa.TEXT(), autoincrement=False, nullable=True),
    sa.ForeignKeyConstraint(['userId'], ['User.id'], name=op.f('PushSubscription_userId_fkey'), onupdate='CASCADE', ondelete='SET NULL'),
    sa.PrimaryKeyConstraint('id', name=op.f('PushSubscription_pkey'))
    )
    op.create_index(op.f('PushSubscription_userId_idx'), 'PushSubscription', ['userId'], unique=False)
    op.create_index(op.f('PushSubscription_endpoint_key'), 'PushSubscription', ['endpoint'], unique=True)
    op.create_index(op.f('PushSubscription_endpoint_idx'), 'PushSubscription', ['endpoint'], unique=False)
    op.create_table('Role',
    sa.Column('id', sa.TEXT(), autoincrement=False, nullable=False),
    sa.Column('name', sa.TEXT(), autoincrement=False, nullable=False),
    sa.Column('description', sa.TEXT(), autoincrement=False, nullable=True),
    sa.Column('createdAt', postgresql.TIMESTAMP(precision=3), server_default=sa.text('CURRENT_TIMESTAMP'), autoincrement=False, nullable=False),
    sa.Column('updatedAt', postgresql.TIMESTAMP(precision=3), autoincrement=False, nullable=False),
    sa.PrimaryKeyConstraint('id', name=op.f('Role_pkey'))
    )
    op.create_index(op.f('Role_name_key'), 'Role', ['name'], unique=True)
    op.create_table('Loan',
    sa.Column('id', sa.VARCHAR(), autoincrement=False, nullable=False),
    sa.Column('customerId', sa.VARCHAR(), autoincrement=False, nullable=False),
    sa.Column('branchId', sa.VARCHAR(), autoincrement=False, nullable=False),
    sa.Column('loanAmount', sa.DOUBLE_PRECISION(precision=53), autoincrement=False, nullable=False),
    sa.Column('interestRate', sa.DOUBLE_PRECISION(precision=53), autoincrement=False, nullable=False),
    sa.Column('termMonths', sa.INTEGER(), autoincrement=False, nullable=False),
    sa.Column('monthlyPayment', sa.DOUBLE_PRECISION(precision=53), autoincrement=False, nullable=True),
    sa.Column('purpose', sa.VARCHAR(), autoincrement=False, nullable=True),
    sa.Column('status', sa.VARCHAR(), autoincrement=False, nullable=False),
    sa.Column('applicationDate', postgresql.TIMESTAMP(), server_default=sa.text('now()'), autoincrement=False, nullable=True),
    sa.Column('startDate', postgresql.TIMESTAMP(), autoincrement=False, nullable=True),
    sa.Column('nextPaymentDate', postgresql.TIMESTAMP(), autoincrement=False, nullable=True),
    sa.Column('remainingBalance', sa.DOUBLE_PRECISION(precision=53), autoincrement=False, nullable=True),
    sa.Column('collateralDescription', sa.VARCHAR(), autoincrement=False, nullable=True),
    sa.Column('repaymentSchedule', postgresql.JSON(astext_type=sa.Text()), autoincrement=False, nullable=True),
    sa.Column('createdAt', postgresql.TIMESTAMP(), server_default=sa.text('now()'), autoincrement=False, nullable=True),
    sa.Column('updatedAt', postgresql.TIMESTAMP(), server_default=sa.text('now()'), autoincrement=False, nullable=True),
    sa.ForeignKeyConstraint(['branchId'], ['Branch.id'], name=op.f('Loan_branchId_fkey')),
    sa.ForeignKeyConstraint(['customerId'], ['User.id'], name=op.f('Loan_customerId_fkey')),
    sa.PrimaryKeyConstraint('id', name=op.f('Loan_pkey'))
    )
    op.create_table('_prisma_migrations',
    sa.Column('id', sa.VARCHAR(length=36), autoincrement=False, nullable=False),
    sa.Column('checksum', sa.VARCHAR(length=64), autoincrement=False, nullable=False),
    sa.Column('finished_at', postgresql.TIMESTAMP(timezone=True), autoincrement=False, nullable=True),
    sa.Column('migration_name', sa.VARCHAR(length=255), autoincrement=False, nullable=False),
    sa.Column('logs', sa.TEXT(), autoincrement=False, nullable=True),
    sa.Column('rolled_back_at', postgresql.TIMESTAMP(timezone=True), autoincrement=False, nullable=True),
    sa.Column('started_at', postgresql.TIMESTAMP(timezone=True), server_default=sa.text('now()'), autoincrement=False, nullable=False),
    sa.Column('applied_steps_count', sa.INTEGER(), server_default=sa.text('0'), autoincrement=False, nullable=False),
    sa.PrimaryKeyConstraint('id', name=op.f('_prisma_migrations_pkey'))
    )
    op.create_table('User',
    sa.Column('id', sa.TEXT(), autoincrement=False, nullable=False),
    sa.Column('email', sa.TEXT(), autoincrement=False, nullable=False),
    sa.Column('name', sa.TEXT(), autoincrement=False, nullable=False),
    sa.Column('password', sa.TEXT(), autoincrement=False, nullable=False),
    sa.Column('role', sa.TEXT(), server_default=sa.text("'user'::text"), autoincrement=False, nullable=False),
    sa.Column('branchId', sa.TEXT(), autoincrement=False, nullable=True),
    sa.Column('isActive', sa.BOOLEAN(), server_default=sa.text('true'), autoincrement=False, nullable=False),
    sa.Column('lastLogin', postgresql.TIMESTAMP(precision=3), autoincrement=False, nullable=True),
    sa.Column('createdAt', postgresql.TIMESTAMP(precision=3), server_default=sa.text('CURRENT_TIMESTAMP'), autoincrement=False, nullable=False),
    sa.Column('updatedAt', postgresql.TIMESTAMP(precision=3), autoincrement=False, nullable=False),
    sa.Column('failedLoginAttempts', sa.INTEGER(), server_default=sa.text('0'), autoincrement=False, nullable=False),
    sa.Column('lockedUntil', postgresql.TIMESTAMP(precision=3), autoincrement=False, nullable=True),
    sa.Column('username', sa.TEXT(), autoincrement=False, nullable=False),
    sa.Column('image', sa.TEXT(), autoincrement=False, nullable=True),
    sa.Column('preferences', postgresql.JSONB(astext_type=sa.Text()), autoincrement=False, nullable=True),
    sa.ForeignKeyConstraint(['branchId'], ['Branch.id'], name='User_branchId_fkey', onupdate='CASCADE', ondelete='SET NULL'),
    sa.PrimaryKeyConstraint('id', name='User_pkey'),
    postgresql_ignore_search_path=False
    )
    op.create_index(op.f('User_username_key'), 'User', ['username'], unique=True)
    op.create_index(op.f('User_email_key'), 'User', ['email'], unique=True)
    op.create_index(op.f('User_branchId_idx'), 'User', ['branchId'], unique=False)
    op.create_table('UserBranchAssignment',
    sa.Column('id', sa.TEXT(), autoincrement=False, nullable=False),
    sa.Column('userId', sa.TEXT(), autoincrement=False, nullable=False),
    sa.Column('branchId', sa.TEXT(), autoincrement=False, nullable=False),
    sa.Column('isDefault', sa.BOOLEAN(), server_default=sa.text('false'), autoincrement=False, nullable=False),
    sa.Column('createdAt', postgresql.TIMESTAMP(precision=3), server_default=sa.text('CURRENT_TIMESTAMP'), autoincrement=False, nullable=False),
    sa.Column('updatedAt', postgresql.TIMESTAMP(precision=3), autoincrement=False, nullable=False),
    sa.ForeignKeyConstraint(['branchId'], ['Branch.id'], name=op.f('UserBranchAssignment_branchId_fkey'), onupdate='CASCADE', ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['userId'], ['User.id'], name=op.f('UserBranchAssignment_userId_fkey'), onupdate='CASCADE', ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('UserBranchAssignment_pkey'))
    )
    op.create_index(op.f('UserBranchAssignment_userId_idx'), 'UserBranchAssignment', ['userId'], unique=False)
    op.create_index(op.f('UserBranchAssignment_userId_branchId_key'), 'UserBranchAssignment', ['userId', 'branchId'], unique=True)
    op.drop_index(op.f('ix_payments_loan_id'), table_name='payments')
    op.drop_index(op.f('ix_payments_collected_by'), table_name='payments')
    op.drop_table('payments')
    op.drop_index(op.f('ix_loans_loan_officer_id'), table_name='loans')
    op.drop_index(op.f('ix_loans_customer_id'), table_name='loans')
    op.drop_index(op.f('ix_loans_branch_id'), table_name='loans')
    op.drop_table('loans')
    op.drop_table('loan_officers')
    op.drop_table('customers')
    op.drop_index(op.f('ix_users_username'), table_name='users')
    op.drop_index(op.f('ix_users_branch_id'), table_name='users')
    op.drop_table('users')
    op.drop_index(op.f('ix_branches_code'), table_name='branches')
    op.drop_table('branches')
    # ### end Alembic commands ###
