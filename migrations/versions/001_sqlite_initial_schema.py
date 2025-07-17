"""SQLite compatible initial database schema

Revision ID: 001_sqlite_initial
Revises: 
Create Date: 2025-01-27 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '001_sqlite_initial'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    
    # Create branches table
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
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=True),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=True),
        sa.ForeignKeyConstraint(['parent_id'], ['branches.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('code')
    )
    
    # Create users table
    op.create_table('users',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('user_type', sa.String(length=20), nullable=False),  # Using String instead of Enum for SQLite
        sa.Column('username', sa.String(length=50), nullable=False),
        sa.Column('email', sa.String(length=100), nullable=False),
        sa.Column('phone_number', sa.String(length=20), nullable=False),
        sa.Column('hashed_password', sa.String(length=255), nullable=False),
        sa.Column('branch_id', sa.String(), nullable=True),
        sa.Column('last_login', sa.DateTime(), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=False),
        sa.Column('is_verified', sa.Boolean(), nullable=False),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=True),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=True),
        sa.ForeignKeyConstraint(['branch_id'], ['branches.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('email'),
        sa.UniqueConstraint('phone_number'),
        sa.UniqueConstraint('username')
    )
    
    # Create customers table
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
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=True),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=True),
        sa.ForeignKeyConstraint(['id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('id_card_number')
    )
    
    # Create loan_officers table
    op.create_table('loan_officers',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('employee_id', sa.String(), nullable=False),
        sa.Column('hire_date', sa.DateTime(), nullable=False),
        sa.Column('department', sa.String(), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=True),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=True),
        sa.ForeignKeyConstraint(['id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('employee_id')
    )
    
    # Create loans table
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
        sa.Column('status', sa.String(length=20), nullable=False),  # Using String instead of Enum for SQLite
        sa.Column('application_date', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=True),
        sa.Column('approval_date', sa.DateTime(), nullable=True),
        sa.Column('start_date', sa.DateTime(), nullable=True),
        sa.Column('next_payment_date', sa.DateTime(), nullable=True),
        sa.Column('end_date', sa.DateTime(), nullable=True),
        sa.Column('remaining_balance', sa.Float(), nullable=True),
        sa.Column('total_paid', sa.Float(), nullable=True),
        sa.Column('collateral_description', sa.String(), nullable=True),
        sa.Column('collateral_value', sa.Float(), nullable=True),
        sa.Column('repayment_schedule', sa.Text(), nullable=True),  # Using Text instead of JSON for SQLite
        sa.Column('notes', sa.String(), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=True),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=True),
        sa.ForeignKeyConstraint(['branch_id'], ['branches.id'], ),
        sa.ForeignKeyConstraint(['customer_id'], ['customers.id'], ),
        sa.ForeignKeyConstraint(['loan_officer_id'], ['loan_officers.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create payments table
    op.create_table('payments',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('loan_id', sa.String(), nullable=False),
        sa.Column('collected_by', sa.String(), nullable=True),
        sa.Column('amount', sa.Numeric(precision=12, scale=2), nullable=False),
        sa.Column('payment_date', sa.DateTime(), nullable=False),
        sa.Column('payment_method', sa.String(length=20), nullable=False),  # Using String instead of Enum for SQLite
        sa.Column('status', sa.String(length=20), nullable=False),  # Using String instead of Enum for SQLite
        sa.Column('reference_number', sa.String(length=50), nullable=True),
        sa.Column('receipt_number', sa.String(length=50), nullable=True),
        sa.Column('transaction_id', sa.String(length=100), nullable=True),
        sa.Column('principal_amount', sa.Numeric(precision=12, scale=2), nullable=False),
        sa.Column('interest_amount', sa.Numeric(precision=12, scale=2), nullable=False),
        sa.Column('late_fee', sa.Numeric(precision=12, scale=2), nullable=True),
        sa.Column('other_fees', sa.Numeric(precision=12, scale=2), nullable=True),
        sa.Column('notes', sa.Text(), nullable=True),
        sa.Column('attachment_url', sa.String(length=255), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=True),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=True),
        sa.ForeignKeyConstraint(['collected_by'], ['users.id'], ),
        sa.ForeignKeyConstraint(['loan_id'], ['loans.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('receipt_number'),
        sa.UniqueConstraint('reference_number')
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table('payments')
    op.drop_table('loans')
    op.drop_table('loan_officers')
    op.drop_table('customers')
    op.drop_table('users')
    op.drop_table('branches')