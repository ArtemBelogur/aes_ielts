from pydantic import BaseModel, EmailStr
from enum import Enum

class UserRole(str, Enum):
    student = "student"
    teacher = "teacher"
    admin = "admin"

class UserCreate(BaseModel):
    email: EmailStr
    password: str
    full_name: str
    role: UserRole

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserOut(BaseModel):
    id: int
    email: EmailStr
    full_name: str
    role: UserRole

    class Config:
        orm_mode = True