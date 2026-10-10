---
title: "Database Connection Pooling: Why APIs Time Out While PostgreSQL Has Capacity"
description: "Why fast SQL can sit behind slow connection checkout, how SQLAlchemy pools scale across workers, and when PgBouncer helps or merely moves the queue."
section: "Blog"
type: "Concept"
language: "en"
translationKey: "database-connection-pooling-why-apis-time-out"
pubDate: 2026-11-25
featured: true
draft: false
---

*An API takes 15 seconds to respond, but its SQL runs in 20 milliseconds. PostgreSQL still has CPU and memory available. Where did the other 14.98 seconds go?*

## 1. Fast SQL, slow API

Imagine a FastAPI endpoint fetching an order by primary key:

```python
@app.get("/orders/{order_id}")
async def get_order(order_id: int):
    return await order_service.get_order(order_id)
```

The SQL is simple and well indexed:

```sql
SELECT * FROM orders WHERE id = 123456;
```

At higher traffic, some requests still finish quickly while others wait several seconds or time out. PostgreSQL CPU looks moderate and recorded queries are fast. Tracing finally shows that requests spend most of their time **waiting to acquire a connection from the application pool**, before the SQL even reaches PostgreSQL.

This is **connection pool exhaustion**. Database query performance and the application's ability to reach the database are different parts of latency.

## 2. What is a connection pool for?

A PostgreSQL connection involves network setup, authentication, session state, and server resources. A regular client connection is normally served by its own PostgreSQL backend process. Creating and destroying a connection for every HTTP request adds overhead.

A pool keeps connections available for reuse. A request checks one out, does database work, then returns it. With SQLAlchemy, calling `close()` on a checked-out connection commonly returns it to the pool rather than physically closing its TCP socket. A pool also **limits concurrent database access**: when all allowed connections are checked out, the next request waits.

That limit can protect PostgreSQL. It can also create a queue in the application even while the database itself has spare capacity.

## 3. What happens when every connection is busy?

Suppose a pool permits five simultaneous checkouts and eight requests need the database. Five proceed; three wait. If a connection is returned soon, one waiter takes it. If none is returned before the configured timeout, the application raises a pool timeout.

```mermaid
flowchart TD
    R["Incoming API requests"] --> P["Application connection pool"]
    P --> C1["Checked-out connection 1"]
    P --> C2["Checked-out connection 2"]
    P --> C3["Checked-out connection 3"]
    C1 --> DB["PostgreSQL"]
    C2 --> DB
    C3 --> DB
    P --> Q["Requests waiting for checkout"]
    Q --> T{"Connection returned before timeout?"}
    T -->|Yes| P
    T -->|No| E["Pool timeout"]
```

PostgreSQL sees the existing backend sessions, **not** all HTTP requests waiting inside SQLAlchemy. A pool timeout differs from a PostgreSQL `statement_timeout`: the first happens while acquiring a connection; the second applies after a statement has reached the database. A connect timeout and an HTTP request timeout are other, separate boundaries.

## 4. Reading `pool_size`, `max_overflow`, and `pool_timeout`

An illustrative SQLAlchemy async engine:

```python
from sqlalchemy.ext.asyncio import create_async_engine

engine = create_async_engine(
    "postgresql+asyncpg://user:password@localhost:5432/shop",
    pool_size=5,
    max_overflow=5,
    pool_timeout=10,
    pool_pre_ping=True,
)
```

Use secrets management for real credentials. For this pool configuration:

| Parameter | Meaning |
| :--- | :--- |
| `pool_size=5` | Up to five connections retained for reuse. |
| `max_overflow=5` | Up to five extra checked-out connections during bursts. |
| `pool_timeout=10` | Wait up to ten seconds for a checkout. |

The maximum simultaneous checkouts from this **one Engine pool** is `5 + 5 = 10`. SQLAlchemy does not create all five persistent connections when the Engine is constructed; it creates them on demand. Overflow connections are normally discarded when returned. An eleventh request needing a connection waits, then may fail after ten seconds.

Increasing both limits to 100 can make the local queue smaller but allows one process to open as many as 200 database connections. It may simply move the bottleneck into PostgreSQL. More connections do not automatically mean more completed queries per second.

## 5. Every worker can have its own pool

Production FastAPI deployments often have multiple processes and replicas. Suppose there are three replicas, four worker processes per replica, and one Engine in each worker with `pool_size=5`, `max_overflow=5`:

```text
3 replicas × 4 workers × (5 + 5) = 120 potential checkouts
```

That is a **configured upper bound**, not 120 connections created immediately or permanently. It also excludes background jobs, migrations, monitoring, and other services. PostgreSQL reserves some connection slots for privileged roles, so applications cannot necessarily use every slot in `max_connections`.

This is why adding workers can make connection errors worse even if each worker's pool looks small. Count pools across the **whole deployment**. Also avoid sharing already-open pooled connections across forked worker processes; create or dispose the Engine/pool appropriately within each process lifecycle.

## 6. Connection leaks and session scope

A pool may run dry because sessions or transactions are not released promptly. This code does not guarantee cleanup:

```python
async def get_order(order_id: int):
    session = SessionLocal()
    result = await session.execute(select(Order).where(Order.id == order_id))
    return result.scalar_one_or_none()
```

Use a context manager for a well-defined lifecycle:

```python
async def get_order(order_id: int):
    async with SessionLocal() as session:
        result = await session.execute(select(Order).where(Order.id == order_id))
        return result.scalar_one_or_none()
```

The session is closed on normal return or exception. But a SQLAlchemy Session is **not automatically a checked-out PostgreSQL connection for its entire lifetime**. It generally acquires one when needed and can return it when its transaction ends. Measure checkout duration and transaction scope; counting Python Session objects is not the same as counting active database connections.

## 7. `idle in transaction`: doing nothing while holding resources

Consider this request:

```python
async with SessionLocal() as session:
    result = await session.execute(select(Order).where(Order.id == order_id))
    await external_service_call()
    return result.scalar_one_or_none()
```

If the query started a transaction and the connection remains checked out while the external call waits, PostgreSQL can show the session as `idle in transaction`. The database is not executing SQL, yet that connection is unavailable to other requests. Long transactions may also retain locks or old row versions and interfere with maintenance.

If the external call is not part of the same required transaction, finish database work and copy the data you need **before** making it:

```python
async def get_order_details(order_id: int):
    async with SessionLocal() as session:
        result = await session.execute(select(Order).where(Order.id == order_id))
        order = result.scalar_one_or_none()
        if order is None:
            return None
        order_data = {"id": order.id, "status": order.status}

    external_data = await external_service_call()
    return {**order_data, "external": external_data}
```

Do not split a transaction when atomic business changes truly require it. FastAPI `yield` dependencies can manage session cleanup, but long-lived or streaming responses require attention to **when** that cleanup happens. Release database resources early once they are no longer needed.

## 8. Measure the missing time

Break an API request into separate phases: request handling, connection checkout, SQL execution, result processing, and response. A synthetic trace might show:

| Phase | Time |
| :--- | ---: |
| Waiting for a connection | 8,200 ms |
| Executing SQL | 25 ms |
| Other processing | 15 ms |
| Total | 8,240 ms |

Neither API latency alone nor SQL execution time alone identifies the queue. Monitor pool checked-out and available counts, acquisition wait, timeout count, and connection hold time. Alongside them, inspect PostgreSQL active sessions, `idle in transaction`, lock waits, and available connection slots.

High checked-out count can be normal if queries are doing useful work. High checkout wait with short SQL time suggests a too-small pool, excessive hold time, or both. Measurements, not a single threshold, distinguish them.

## 9. What PostgreSQL can tell us

`pg_stat_activity` summarizes existing server sessions:

```sql
SELECT application_name, state, COUNT(*) AS connection_count
FROM pg_stat_activity
WHERE datname = current_database()
GROUP BY application_name, state
ORDER BY connection_count DESC;
```

`active` means executing; `idle` means waiting for a new command; `idle in transaction` means a transaction is open without an executing statement. A connection pooled for reuse can legitimately look `idle`; that alone does **not** prove a leak.

To look for long-running transactions and waits:

```sql
SELECT pid, application_name, state,
       NOW() - xact_start AS transaction_age,
       wait_event_type, wait_event
FROM pg_stat_activity
WHERE xact_start IS NOT NULL
ORDER BY transaction_age DESC;
```

Visibility depends on monitoring privileges. And `pg_stat_activity` cannot reveal how many HTTP requests are still waiting **inside the application pool**. Instrument both sides.

## 10. Where PgBouncer fits

With many application replicas and background workers, even sensible per-process pools may produce too many PostgreSQL server sessions. **PgBouncer** sits between clients and PostgreSQL and can reuse a controlled number of server connections:

```mermaid
flowchart TD
    LB["Load balancer"] --> A1["App replica 1"]
    LB --> A2["App replica 2"]
    LB --> A3["App replica 3"]
    A1 --> P1["App pool"]
    A2 --> P2["App pool"]
    A3 --> P3["App pool"]
    P1 --> PG["PgBouncer"]
    P2 --> PG
    P3 --> PG
    PG --> SP["Controlled server connections"]
    SP --> DB["PostgreSQL"]
```

It does **not** make PostgreSQL capable of unlimited concurrent work. If queries or transactions stay slow, a queue may move from the application to PgBouncer. Some deployments use a modest application pool in front of it; others minimize client-side pooling. The right combination depends on driver behavior, PgBouncer mode, and workload. If both layers queue, observe both layers.

## 11. Session versus transaction pooling

In **session pooling**, a client holds one server connection for its entire client session. This preserves session-level state but may leave less opportunity to share server connections among long-lived clients.

In **transaction pooling**, a server connection is assigned for a transaction and can be reused by another client afterward. This can reduce the number of server connections needed when transactions are short. But two transactions from one client may run on different PostgreSQL backend sessions.

| Feature | Session pooling | Transaction pooling |
| :--- | :--- | :--- |
| Ordinary transactions | Supported | Supported |
| Session-level advisory locks | Supported | Not compatible |
| `LISTEN` | Supported | Not compatible |
| Arbitrary session `SET/RESET` state | Supported | Not preserved as expected |
| Temporary tables surviving a transaction | Supported | Often incompatible |
| Protocol-level named prepared plans | Supported | Conditional on configuration |

PgBouncer can track protocol-level named prepared statements in transaction pooling when `max_prepared_statements` is nonzero. That does **not** mean SQL `PREPARE` or every driver-side statement cache and session feature will work unchanged. Verify the specific PgBouncer version, driver, and prepared-statement behavior before switching modes.

## 12. Monitor PgBouncer too

Run `SHOW POOLS;` and `SHOW STATS;` on the **PgBouncer admin console**, not on the application database as ordinary SQL. Useful pool fields include `cl_waiting` (clients waiting for a server connection), `sv_active`, `sv_idle`, and `maxwait`.

Rising `cl_waiting` may indicate a pool limit, long transactions, slow SQL, or database contention. Increasing the PgBouncer pool without understanding the cause can just overload PostgreSQL. Trace the full path: **HTTP request -> application pool -> PgBouncer -> PostgreSQL**.

## 13. Pool sizing is a workload decision

The useful pool size depends on replicas, processes, requests that actually need the database, checkout duration, SQL duration, contention, and other database clients. **HTTP concurrency is not database concurrency.** Two hundred in-flight requests do not necessarily need two hundred simultaneous database connections if many are waiting on external services or processing in memory.

Shorter connection hold time can lower the required pool size. A larger pool can raise database concurrency until CPU, I/O, memory, or locks become contested. Slower queries then hold connections longer, worsening the queue. A pool provides **backpressure**, but an unbounded queue simply turns overload into bad latency. Use finite waits and an application response appropriate to the service's capacity and SLA.

## 14. Benchmark the queue, not just the query

Test increasing load under comparable datasets, SQL, worker counts, and cache conditions. For example, compare a small application pool (`pool_size=2`, no overflow), a somewhat larger pool (`pool_size=5`, `max_overflow=2`), and an appropriate PgBouncer setup. These are **test configurations**, not production recommendations.

Measure requests per second; p50/p95/p99 API latency; connection acquisition and hold time; pool timeouts; SQL time; PostgreSQL active sessions, CPU, I/O and lock waits; and PgBouncer `cl_waiting` if present. Repeat runs and inspect tail latency, not only averages.

Test failure modes too: an unclosed session, a long external call while a transaction is open, and a streaming response. If fixing the lifecycle cuts checkout wait while SQL time stays almost unchanged, that is strong evidence about where the bottleneck was.

## 15. A production debugging sequence

1. Identify whether the error is a pool, connection, statement, lock, or HTTP timeout.
2. Measure connection acquisition separately from SQL execution.
3. Inspect `pg_stat_activity`, transaction age, and wait events.
4. Count all replicas, workers, Engines, pools, and other database clients.
5. Look for leaked sessions and database work held across network calls.
6. If connections are held by slow SQL, investigate plans and lock contention.
7. Only then tune pool limits, application concurrency, or PgBouncer, and benchmark again.
8. After deployment, monitor checkout wait, utilization, error rate, and p95/p99 latency.

This avoids treating a symptom at one layer by overloading the next.

## 16. Common mistakes

- Raising PostgreSQL `max_connections` immediately without understanding why existing slots are occupied.
- Choosing a pool size for one worker and forgetting it multiplies across processes and replicas.
- Leaving sessions or transactions open, or holding a connection during an external API call.
- Calling every PostgreSQL `idle` session a leak, although a reusable pooled connection can be idle.
- Using PgBouncer transaction pooling without auditing session-dependent features.
- Measuring SQL time but not checkout time.
- Assuming async I/O gives PostgreSQL unlimited connections or query capacity.

No one pool setting is optimal for every deployment. Tune from measured workload and connection lifecycle.

## 17. The missing part of API latency

Connection pooling reuses expensive connections and controls concurrency. It can also create a hidden queue. SQLAlchemy's pool, PgBouncer, and PostgreSQL each have their own limits and signals. A 20 ms query does not imply a 20 ms database phase when the request waited seconds for a connection.

**A good pool is not the largest pool. It supplies enough connections for the real workload while limiting concurrency so the database remains stable.** Keep transactions short, return resources promptly, and measure the wait before changing the limit.

## References and further reading

- [SQLAlchemy: Connection Pooling](https://docs.sqlalchemy.org/en/20/core/pooling.html)
- [SQLAlchemy: Error Messages](https://docs.sqlalchemy.org/en/20/errors.html)
- [SQLAlchemy: Session Basics](https://docs.sqlalchemy.org/en/20/orm/session_basics.html)
- [SQLAlchemy: Pool Events](https://docs.sqlalchemy.org/en/20/core/events.html)
- [FastAPI: Dependencies with Yield](https://fastapi.tiangolo.com/tutorial/dependencies/dependencies-with-yield/)
- [FastAPI: Advanced Dependencies](https://fastapi.tiangolo.com/advanced/advanced-dependencies/)
- [PostgreSQL: Connections and Authentication](https://www.postgresql.org/docs/current/runtime-config-connection.html)
- [PostgreSQL: Monitoring Statistics](https://www.postgresql.org/docs/current/monitoring-stats.html)
- [PostgreSQL: Client Connection Defaults](https://www.postgresql.org/docs/current/runtime-config-client.html)
- [PgBouncer: Features](https://www.pgbouncer.org/features.html)
- [PgBouncer: Configuration](https://www.pgbouncer.org/config)
- [PgBouncer: Usage](https://www.pgbouncer.org/usage)
