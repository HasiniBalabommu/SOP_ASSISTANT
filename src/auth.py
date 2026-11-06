"""
auth.py — simple role-based access control (RBAC).
For the MVP this is intentionally lightweight: no password hashing or
sessions yet, just a role -> permissions model the app layer checks
before allowing an action. Can be upgraded to real authentication
(e.g. with a login form + hashed passwords) later without changing
how the rest of the system calls it.
"""

from enum import Enum


class Role(str, Enum):
    ADMIN = "admin"
    ENGINEER = "engineer"
    TECHNICIAN = "technician"


# Which actions each role is allowed to perform
ROLE_PERMISSIONS: dict[Role, set[str]] = {
    Role.ADMIN: {"upload_documents", "delete_documents", "ask_questions", "manage_users", "view_all_citations"},
    Role.ENGINEER: {"upload_documents", "ask_questions", "view_all_citations"},
    Role.TECHNICIAN: {"ask_questions", "view_all_citations"},
}


class User:
    def __init__(self, username: str, role: Role):
        self.username = username
        self.role = role

    def can(self, action: str) -> bool:
        return action in ROLE_PERMISSIONS.get(self.role, set())

    def __repr__(self):
        return f"User(username={self.username!r}, role={self.role.value})"


def require_permission(user: User, action: str) -> None:
    """Raises PermissionError if the user isn't allowed to perform this action."""
    if not user.can(action):
        raise PermissionError(f"User '{user.username}' ({user.role.value}) is not allowed to '{action}'")


if __name__ == "__main__":
    admin = User("alice", Role.ADMIN)
    technician = User("bob", Role.TECHNICIAN)

    print(admin, "-> can upload_documents:", admin.can("upload_documents"))
    print(technician, "-> can upload_documents:", technician.can("upload_documents"))
    print(technician, "-> can ask_questions:", technician.can("ask_questions"))

    try:
        require_permission(technician, "delete_documents")
    except PermissionError as e:
        print(f"\nBlocked as expected: {e}")