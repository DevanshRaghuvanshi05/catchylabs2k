
byok_endpoints = '''

# =============================================================================
# BYOK (Bring Your Own API Key) Credential Endpoints
# =============================================================================
# These endpoints let authenticated users submit, view, and delete their own
# Gemini and SerpAPI keys.  Keys are encrypted at rest (AES-256-GCM) and
# stored per-user in the user_api_credentials table.
# Auth: Supabase bearer token (Authorization: Bearer <access_token>)
# =============================================================================


class CredentialSubmission(BaseModel):
    """Request body for submitting an API key."""
    api_key: str = Field(..., min_length=10, description="Raw API key (never stored in plaintext).")


async def _require_supabase_user(authorization: Optional[str] = Header(default=None)) -> dict:
    """
    Dependency: validates Supabase JWT and returns the authenticated user dict.
    Raises HTTP 401 if the token is missing or invalid.
    """
    from provider_auth import verify_supabase_access_token
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid Authorization header. Expected: Bearer <supabase_access_token>",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return await verify_supabase_access_token(authorization[7:])


@app.post(
    "/api/credentials/gemini",
    response_model=Dict[str, Any],
    status_code=status.HTTP_200_OK,
    summary="Submit Gemini API Key",
    description=(
        "Validates and securely stores the user\'s Gemini API key (encrypted at rest). "
        "Requires a valid Supabase session token in the Authorization header. "
        "Returns only the masked key — the plaintext key is never returned."
    ),
)
async def submit_gemini_credential(
    body: CredentialSubmission,
    user: dict = Depends(_require_supabase_user),
) -> Dict[str, Any]:
    from services.credential_manager import store_credential, validate_gemini_key
    user_id = user["id"]
    logger.info(f"[BYOK] POST /api/credentials/gemini for user {user_id[:8]}...")
    # Validate before storing (attempt a live API call)
    is_valid = validate_gemini_key(body.api_key)
    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Gemini API key validation failed. Please check the key and try again.",
        )
    result = store_credential(user_id, "gemini", body.api_key)
    return {"success": True, "credential": result}


@app.post(
    "/api/credentials/serpapi",
    response_model=Dict[str, Any],
    status_code=status.HTTP_200_OK,
    summary="Submit SerpAPI Key",
    description=(
        "Validates and securely stores the user\'s SerpAPI key (encrypted at rest). "
        "Requires a valid Supabase session token. Returns only the masked key."
    ),
)
async def submit_serpapi_credential(
    body: CredentialSubmission,
    user: dict = Depends(_require_supabase_user),
) -> Dict[str, Any]:
    from services.credential_manager import store_credential, validate_serpapi_key
    user_id = user["id"]
    logger.info(f"[BYOK] POST /api/credentials/serpapi for user {user_id[:8]}...")
    is_valid = validate_serpapi_key(body.api_key)
    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="SerpAPI key validation failed. Please check the key and try again.",
        )
    result = store_credential(user_id, "serpapi", body.api_key)
    return {"success": True, "credential": result}


@app.get(
    "/api/credentials/status",
    response_model=Dict[str, Any],
    status_code=status.HTTP_200_OK,
    summary="Credential Status",
    description=(
        "Returns the masked configuration status for the authenticated user\'s Gemini and SerpAPI keys. "
        "Never returns plaintext keys."
    ),
)
async def get_credentials_status(
    user: dict = Depends(_require_supabase_user),
) -> Dict[str, Any]:
    from services.credential_manager import get_credential_status
    user_id = user["id"]
    return {"user_id": user_id, "credentials": get_credential_status(user_id)}


@app.delete(
    "/api/credentials/gemini",
    response_model=Dict[str, Any],
    status_code=status.HTTP_200_OK,
    summary="Delete Gemini API Key",
    description="Deletes the authenticated user\'s stored Gemini API key.",
)
async def delete_gemini_credential(
    user: dict = Depends(_require_supabase_user),
) -> Dict[str, Any]:
    from services.credential_manager import delete_credential
    user_id = user["id"]
    deleted = delete_credential(user_id, "gemini")
    logger.info(f"[BYOK] DELETE /api/credentials/gemini for user {user_id[:8]}... deleted={deleted}")
    return {"success": True, "deleted": deleted, "provider": "gemini"}


@app.delete(
    "/api/credentials/serpapi",
    response_model=Dict[str, Any],
    status_code=status.HTTP_200_OK,
    summary="Delete SerpAPI Key",
    description="Deletes the authenticated user\'s stored SerpAPI key.",
)
async def delete_serpapi_credential(
    user: dict = Depends(_require_supabase_user),
) -> Dict[str, Any]:
    from services.credential_manager import delete_credential
    user_id = user["id"]
    deleted = delete_credential(user_id, "serpapi")
    logger.info(f"[BYOK] DELETE /api/credentials/serpapi for user {user_id[:8]}... deleted={deleted}")
    return {"success": True, "deleted": deleted, "provider": "serpapi"}
'''

with open('app.py', 'a', encoding='utf-8') as f:
    f.write(byok_endpoints)
print('BYOK endpoints appended to app.py')
