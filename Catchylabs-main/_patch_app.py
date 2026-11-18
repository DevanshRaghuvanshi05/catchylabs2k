
content = open('app.py', 'r', encoding='utf-8').read()

old = 'def create_research_report(brief: ResearchBrief) -> Dict[str, Any]:'
new = '''def create_research_report(
    brief: ResearchBrief,
    authorization: Optional[str] = Header(default=None),
) -> Dict[str, Any]:'''

if old in content:
    content = content.replace(old, new)
    print('Signature patched OK')
else:
    print('SIGNATURE NOT FOUND')
    idx = content.find('def create_research_report')
    print(repr(content[idx:idx+100]))

old2 = '''    logger.info(f"Received HTTP POST /research request for Brand={brief.brand}, Market={brief.market}")
    brief_dict = brief.model_dump()

    # 1. Daily rate-limit check'''

new2 = '''    logger.info(f"Received HTTP POST /research request for Brand={brief.brand}, Market={brief.market}")
    brief_dict = brief.model_dump()
    # Inject user_id from Supabase JWT (optional)
    if authorization and authorization.startswith("Bearer "):
        import asyncio
        from provider_auth import verify_supabase_access_token
        try:
            loop = asyncio.new_event_loop()
            user = loop.run_until_complete(verify_supabase_access_token(authorization[7:]))
            loop.close()
            brief_dict["user_id"] = user.get("id")
        except Exception:
            pass  # Non-authenticated runs get no user_id

    # 1. Daily rate-limit check'''

if old2 in content:
    content = content.replace(old2, new2)
    print('Body patched OK')
else:
    print('BODY PATTERN NOT FOUND')

open('app.py', 'w', encoding='utf-8').write(content)
print('Done')
