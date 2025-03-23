from datetime import datetime, timezone
from enum import Enum
from typing import List

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)


class Token(Base):
    __tablename__ = "tokens"

    created: Mapped[datetime] = mapped_column(
        default=lambda: datetime.now(timezone.utc)
    )
    token: Mapped[str] = mapped_column(unique=True, index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    user: Mapped["User"] = relationship(back_populates="tokens")


class CompanyType(Base):
    __tablename__ = "company_types"
    name: Mapped[str]
    companies: Mapped[List["Company"]] = relationship(back_populates="company_type")

    def __repr__(self):
        return f"<CompanyType={self.name}>"


class Company(Base):
    __tablename__ = "companies"
    name: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)
    postal_code: Mapped[str]
    email: Mapped[str] = mapped_column(String(1000))
    description: Mapped[str] = mapped_column(Text)
    analytics_module: Mapped[bool] = mapped_column(nullable=True)
    # New
    website: Mapped[str] = mapped_column(nullable=True)

    # Relationships
    company_type: Mapped["CompanyType"] = relationship(back_populates="companies")
    company_type_id: Mapped[int] = mapped_column(
        ForeignKey("company_types.id", ondelete="SET NULL"), nullable=True
    )
    users: Mapped[List["User"]] = relationship(back_populates="company")

    def __repr__(self):
        return f"<Company={self.name}>"


class EnrollmentStatus(str, Enum):
    """Status of a user's enrollment in a course"""

    ENROLLED = "enrolled"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    DROPPED = "dropped"


class Conversation(Base):
    """Stores conversation history for users"""
    __tablename__ = "conversations"
    
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    
    # Relationships
    user: Mapped["User"] = relationship(back_populates="conversations")
    messages: Mapped[List["ConversationMessage"]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="ConversationMessage.created_at"
    )


class ConversationMessage(Base):
    """Stores individual messages within a conversation"""
    __tablename__ = "conversation_messages"
    
    text: Mapped[str] = mapped_column(Text, nullable=False)
    is_user: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    conversation_id: Mapped[int] = mapped_column(ForeignKey("conversations.id"))
    
    # Relationships
    conversation: Mapped["Conversation"] = relationship(back_populates="messages")


class User(Base):
    """User model representing system users who can be enrolled in courses and belong to companies"""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    first_name: Mapped[str] = mapped_column(String(100), index=True)
    last_name: Mapped[str] = mapped_column(String(100), index=True)
    disabled: Mapped[bool] = mapped_column(Boolean, default=False)
    hashed_password: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(
        server_default=func.now()  # Uses database server time
    )

    # Relationships
    company_id: Mapped[int | None] = mapped_column(
        ForeignKey("companies.id"), nullable=True
    )
    company: Mapped["Company | None"] = relationship(back_populates="users")
    course_enrollments: Mapped[List["UserCourseEnrollment"]] = relationship(
        back_populates="user"
    )
    conversations: Mapped[List["Conversation"]] = relationship(
        back_populates="user", 
        cascade="all, delete-orphan",
        order_by="Conversation.created_at.desc()"
    )

    # Auth
    is_superuser: Mapped[bool] = mapped_column(Boolean, default=False)
    tokens: Mapped[list["Token"]] = relationship(back_populates="user")

    @property
    def full_name(self) -> str:
        return f"{self.first_name} {self.last_name}"

    def __repr__(self) -> str:
        return f"<User {self.email} ({self.full_name})>"


class Course(Base):
    """Course model representing available courses in the system"""

    __tablename__ = "courses"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200), unique=True, index=True)
    description: Mapped[str | None] = mapped_column(String(1000), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        server_default=func.now()  # Uses database server time
    )
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    # Relationships
    student_enrollments: Mapped[List["UserCourseEnrollment"]] = relationship(
        back_populates="course"
    )

    def __repr__(self) -> str:
        return f"<Course {self.name}>"


class UserCourseEnrollment(Base):
    """Association model tracking user enrollments in courses"""

    __tablename__ = "user_course_enrollments"
    __table_args__ = (
        UniqueConstraint("user_id", "course_id", name="uix_user_course_enrollment"),
    )

    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), primary_key=True)
    course_id: Mapped[int] = mapped_column(ForeignKey("courses.id"), primary_key=True)
    enrolled_at: Mapped[datetime] = mapped_column(
        server_default=func.now()  # Uses database server time
    )
    status: Mapped[EnrollmentStatus] = mapped_column(default=EnrollmentStatus.ENROLLED)
    completion_date: Mapped[datetime | None] = mapped_column(nullable=True)
    grade: Mapped[float | None] = mapped_column(
        nullable=True, comment="Final grade for the course (0-100)"
    )

    # Relationships
    user: Mapped["User"] = relationship(back_populates="course_enrollments")
    course: Mapped["Course"] = relationship(back_populates="student_enrollments")

    def __repr__(self) -> str:
        return f"<UserCourseEnrollment user_id={self.user_id} course_id={self.course_id} status={self.status.value}>"
