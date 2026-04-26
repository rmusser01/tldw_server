# Evaluations Module Security Guide

## Overview
This document outlines the security measures, best practices, and potential risks in the Evaluations module.

## Security Measures Implemented

### 1. Path Traversal Protection
- **Location**: `evaluation_manager.py:_get_db_path()`
- **Implementation**:
  - Sanitizes user-provided paths by removing `..` sequences
  - Normalizes paths using `os.path.normpath()`
  - Validates that resolved paths stay within project boundaries
  - Falls back to safe default path if validation fails
- **Testing**: `test_security.py:TestPathTraversalProtection`

### 2. SQL Injection Prevention
- **Implementation**:
  - All database queries use parameterized statements
  - No string concatenation for SQL queries with user input
  - Prepared statements for all data operations
- **Example**:
  ```python
  conn.execute("INSERT INTO table (col) VALUES (?)", (user_input,))
  ```
- **Testing**: `test_security.py:TestSQLInjectionPrevention`

### 3. Webhook Security (SSRF Protection)
- **Location**: `webhook_security.py`
- **Measures**:
  - Blocks private IP addresses (RFC 1918, loopback, link-local)
  - Validates against blocked ports (SSH, database ports, etc.)
  - DNS resolution validation
  - Domain allowlist/blocklist support
  - SSL certificate validation
- **Limitations**:
  - DNS rebinding attacks need additional follow-redirect validation
  - Response size limits should be implemented
- **Testing**: `test_security.py:TestWebhookSecurityValidation`

### 4. Thread Safety
- **Location**: `connection_pool.py`
- **Implementation**:
  - SQLite connections use `check_same_thread=True`
  - Thread-local connection management
  - Proper locking for pool operations
  - Connection lifecycle management
- **Testing**: `test_security.py:TestConnectionPoolThreadSafety`

### 5. Input Validation
- **Score Parsing** (`evaluation_manager.py`):
  - Strict validation of score ranges (0-10)
  - JSON parsing with type checking
  - Fallback regex patterns with validation
  - Default safe values for parse failures
- **Data Size Limits**:
  - Should implement maximum sizes for metadata
  - Truncation of overly long strings
- **Testing**: `test_security.py:TestScoreParsingSecurit

y`

### 6. Authentication & Authorization
- **Location**: `evals_openai.py`
- **Implementation**:
  - JWT token validation for multi-user mode
  - API key validation for single-user mode
  - Rate limiting per user/IP
  - Canonical evaluations identity mapping for ownership, DB routing, and webhook subjects
- **Considerations**:
  - Implement key rotation mechanism
  - Add request signing for webhooks

### 7. Canonical Identity Handling
- **Location**: `identity.py`, evaluations route modules, Jobs workers
- **Rules**:
  - Resolve one canonical `EvaluationIdentity` from the authenticated user for each request or job.
  - Use `user_scope` for per-user service binding and database path selection.
  - Use `created_by` for ownership checks, row filters, and idempotency records.
  - Use `webhook_user_id` for webhook manager ownership.
- **Do Not**:
  - Do not use the raw `verify_api_key()` return value as an ownership key, limiter subject, or storage-routing identifier.
  - Do not mix route-local `current_user.id`, token strings, and webhook prefixes in separate ad hoc normalization paths.
- **Reason**:
  - The auth artifact can differ from the durable evaluations owner string. Using one canonical identity prevents cross-user leakage, idempotency drift, and service cache misrouting.

## Known Security Risks & Mitigations

### High Priority Risks

1. **Database Migration Failures**
   - **Risk**: Fallback schema creation in production
   - **Mitigation**: Fail fast in production environments
   - **Status**: ✅ Fixed - Production environments now fail on migration errors

2. **Path Traversal in Configuration**
   - **Risk**: User-controlled paths in config could escape boundaries
   - **Mitigation**: Path sanitization and validation
   - **Status**: ✅ Fixed - Paths are now sanitized and validated

3. **Webhook SSRF**
   - **Risk**: Server-side request forgery to internal services
   - **Mitigation**: IP validation, port blocking, DNS validation
   - **Status**: ⚠️ Partial - Needs follow-redirect validation

### Medium Priority Risks

1. **Rate Limiting Bypass**
   - **Risk**: IP-based rate limiting can be bypassed with proxies
   - **Mitigation**: Implement user-based rate limiting
   - **Status**: ⚠️ Needs improvement

2. **Large Payload Attacks**
   - **Risk**: DoS through large metadata or webhook responses
   - **Mitigation**: Implement size limits
   - **Status**: ❌ Not implemented

3. **Concurrent Access Issues**
   - **Risk**: Race conditions in evaluation processing
   - **Mitigation**: Proper transaction isolation
   - **Status**: ✅ Fixed - Thread safety implemented

## Security Best Practices

### For Developers

1. **Database Operations**
   ```python
   # Good - Parameterized query
   conn.execute("SELECT * FROM table WHERE id = ?", (user_id,))

   # Bad - String concatenation
   conn.execute(f"SELECT * FROM table WHERE id = '{user_id}'")
   ```

2. **Path Handling**
   ```python
   # Good - Validate and sanitize
   path = Path(user_input).resolve()
   if not path.is_relative_to(safe_directory):
       raise ValueError("Invalid path")

   # Bad - Direct usage
   path = Path(user_input)
   ```

3. **Input Validation**
   ```python
   # Good - Validate type and range
   if isinstance(score, (int, float)) and 0 <= score <= 10:
       validated_score = float(score)

   # Bad - Trust user input
   score = float(user_input)
   ```

### For Deployment

1. **Environment Variables**
   - Set `ENVIRONMENT=production` for production deployments
   - Use strong, randomly generated API keys
   - Enable SSL/TLS for all endpoints

2. **Database Security**
   - Regular backups
   - Encrypted storage
   - Restricted file permissions
   - WAL mode for better concurrency

3. **Monitoring**
   - Log all authentication failures
   - Monitor rate limit violations
   - Track webhook validation failures
   - Alert on unusual patterns

## Security Checklist

### Pre-Production Checklist
- [ ] All critical security fixes applied
- [ ] Security tests passing
- [ ] Rate limiting configured
- [ ] Webhook domain allowlist configured
- [ ] SSL/TLS enabled
- [ ] API keys rotated
- [ ] Database migrations tested
- [ ] Error messages don't leak sensitive info
- [ ] Logging configured (without secrets)
- [ ] Input size limits implemented

### Periodic Security Review
- [ ] Review webhook registrations
- [ ] Audit user permissions
- [ ] Check for unusual API usage patterns
- [ ] Update dependencies for security patches
- [ ] Review and rotate API keys
- [ ] Test backup and recovery procedures

## Incident Response

### If a Security Issue is Discovered

1. **Immediate Actions**
   - Disable affected endpoints if critical
   - Rotate compromised keys
   - Review logs for exploitation attempts

2. **Investigation**
   - Identify scope of issue
   - Check for data exposure
   - Document timeline

3. **Remediation**
   - Apply security patch
   - Test thoroughly
   - Deploy with monitoring

4. **Post-Incident**
   - Update security tests
   - Document lessons learned
   - Review similar code for issues

## Contact

For security concerns or to report vulnerabilities:
- Create a private security advisory on GitHub
- Or contact the security team directly

## Version History

- v1.0.0 - Initial security review and fixes
- v1.0.1 - Added path traversal protection
- v1.0.2 - Fixed thread safety issues
- v1.0.3 - Enhanced score parsing security
