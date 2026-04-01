# Worker API Contract

**Version**: 1.0 | **Date**: 2026-04-02

## WebSocket Endpoint: `/api/chat`

Existing endpoint, unchanged protocol. Serves one voice session per
connection.

### Query Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| voice_prompt | string | yes | Voice prompt filename |
| text_prompt | string | yes | System text prompt (can be empty) |
| seed | int | no | Random seed (-1 or omit for random) |

### Binary Message Protocol (unchanged)

| Kind byte | Direction | Payload |
|-----------|-----------|---------|
| 0x00 | server→client | Handshake (session ready) |
| 0x01 | bidirectional | Opus audio frame |
| 0x02 | server→client | UTF-8 text token |

### Connection Lifecycle

1. Client opens WebSocket to `/api/chat?voice_prompt=X&text_prompt=Y`
2. Server allocates a session slot (or rejects with 503 if at capacity)
3. Server processes system prompts (voice + text)
4. Server sends `0x00` handshake byte
5. Bidirectional audio streaming begins
6. Either side closes the WebSocket to end the session

### Error Responses

| Code | Condition |
|------|-----------|
| 503 Service Unavailable | No session slots available |
| 400 Bad Request | Missing or invalid voice prompt |

---

## HTTP Endpoint: `GET /health`

**New endpoint** on each worker process.

### Response (200 OK)

```json
{
  "worker_id": "worker-1",
  "active_sessions": 2,
  "max_sessions": 5,
  "healthy": true
}
```

### Error Response (503)

Returned when the worker is in a degraded state (e.g., model failed
to load).

---

## HTTP Endpoint: `GET /metrics`

**New endpoint** serving Prometheus-compatible metrics.

### Response (200 OK, text/plain)

```
# HELP personaplex_active_sessions Current active session count
# TYPE personaplex_active_sessions gauge
personaplex_active_sessions 2

# HELP personaplex_session_duration_seconds Session duration
# TYPE personaplex_session_duration_seconds histogram
personaplex_session_duration_seconds_bucket{le="10"} 5
personaplex_session_duration_seconds_bucket{le="60"} 12
personaplex_session_duration_seconds_bucket{le="300"} 15
personaplex_session_duration_seconds_bucket{le="+Inf"} 15
personaplex_session_duration_seconds_count 15
personaplex_session_duration_seconds_sum 1234.5

# HELP personaplex_connection_rejections_total Rejected connections
# TYPE personaplex_connection_rejections_total counter
personaplex_connection_rejections_total 3
```
