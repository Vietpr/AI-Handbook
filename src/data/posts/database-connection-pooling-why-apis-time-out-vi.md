---
title: "Database Connection Pooling: Vì sao API timeout dù PostgreSQL vẫn còn tài nguyên?"
description: "SQL chạy nhanh nhưng API vẫn chậm: tìm thời gian chờ connection, tính pool trên toàn bộ workers và hiểu khi nào PgBouncer thực sự có ích."
section: "Blog"
type: "Concept"
language: "vi"
translationKey: "database-connection-pooling-why-apis-time-out"
pubDate: 2026-11-25
featured: true
draft: false
---

*API mất 15 giây mới trả lời, nhưng câu SQL chỉ chạy 20 mili giây. PostgreSQL vẫn còn CPU và bộ nhớ. Vậy 14,98 giây còn lại nằm ở đâu?*

## 1. SQL nhanh, API vẫn chậm

Hãy tưởng tượng một endpoint FastAPI lấy đơn hàng theo khóa chính:

```python
@app.get("/orders/{order_id}")
async def get_order(order_id: int):
    return await order_service.get_order(order_id)
```

Câu SQL đơn giản và đã có index phù hợp:

```sql
SELECT * FROM orders WHERE id = 123456;
```

Khi lưu lượng tăng, một số request vẫn hoàn thành nhanh, còn số khác chờ nhiều giây hoặc timeout. CPU PostgreSQL chưa cao và những query được ghi nhận đều chạy nhanh. Nhưng trace cho thấy phần lớn thời gian request dành để **chờ lấy một connection từ pool của ứng dụng**, trước cả khi SQL tới PostgreSQL.

Đây là tình trạng **connection pool exhaustion**: các connection có thể sử dụng trong pool đều đang bận. Tốc độ thực thi SQL và khả năng ứng dụng lấy được connection để gửi SQL là hai phần khác nhau của latency.

## 2. Vì sao cần connection pool?

Một connection PostgreSQL cần thiết lập mạng, xác thực, tạo session và sử dụng tài nguyên phía server. Một client connection thông thường được phục vụ bởi một backend process riêng của PostgreSQL. Nếu mỗi HTTP request đều tạo rồi đóng connection mới, chi phí này sẽ lặp lại liên tục.

Pool giữ các connection để tái sử dụng. Request mượn một connection, làm việc với database rồi trả lại. Với SQLAlchemy, gọi `close()` trên một connection đã mượn thường là trả nó về pool, chứ không phải đóng hẳn TCP socket. Pool còn **giới hạn số tác vụ truy cập database đồng thời**: khi tất cả connection được phép dùng đều đã được mượn, request tiếp theo phải chờ.

Giới hạn đó có thể bảo vệ PostgreSQL. Nhưng nó cũng có thể tạo hàng đợi ngay trong ứng dụng, dù database vẫn còn tài nguyên.

## 3. Điều gì xảy ra khi mọi connection đều bận?

Giả sử pool cho phép năm connection được mượn cùng lúc, trong khi tám request cần truy cập database. Năm request đi tiếp; ba request chờ. Nếu có connection được trả kịp, một request đang chờ sẽ lấy nó. Nếu không, ứng dụng báo pool timeout.

```mermaid
flowchart TD
    R["API requests đi vào"] --> P["Connection pool của ứng dụng"]
    P --> C1["Connection đang được mượn 1"]
    P --> C2["Connection đang được mượn 2"]
    P --> C3["Connection đang được mượn 3"]
    C1 --> DB["PostgreSQL"]
    C2 --> DB
    C3 --> DB
    P --> Q["Requests đang chờ lấy connection"]
    Q --> T{"Connection được trả trước timeout?"}
    T -->|Có| P
    T -->|Không| E["Pool timeout"]
```

PostgreSQL thấy các server session hiện có, nhưng **không thấy** mọi HTTP request đang xếp hàng bên trong SQLAlchemy. Pool timeout khác PostgreSQL `statement_timeout`: lỗi thứ nhất xảy ra khi chưa lấy được connection; lỗi thứ hai áp dụng khi câu lệnh đã tới database. Connect timeout và HTTP request timeout cũng là những giới hạn riêng.

## 4. Đọc `pool_size`, `max_overflow` và `pool_timeout`

Ví dụ một SQLAlchemy async engine:

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

Trong hệ thống thật, hãy quản lý thông tin đăng nhập bằng cơ chế secrets. Với cấu hình trên:

| Tham số | Ý nghĩa |
| :--- | :--- |
| `pool_size=5` | Giữ tối đa năm connection trong pool để tái sử dụng. |
| `max_overflow=5` | Cho phép thêm tối đa năm connection được mượn khi có đợt tăng tải. |
| `pool_timeout=10` | Chờ tối đa mười giây để lấy được connection. |

Số connection có thể được mượn đồng thời từ **một Engine pool** tối đa là `5 + 5 = 10`. SQLAlchemy không tạo sẵn cả năm connection khi khởi tạo Engine; chúng được tạo khi cần. Các overflow connection thường bị đóng khi trả về. Request thứ mười một cần database sẽ phải chờ và có thể lỗi sau mười giây.

Tăng cả hai giới hạn lên 100 có thể làm hàng đợi tại ứng dụng ngắn lại, nhưng cũng cho phép một process mở tới 200 database connection. Nút thắt có thể chỉ chuyển sang PostgreSQL. Nhiều connection hơn không tự động tạo ra nhiều query hoàn thành mỗi giây hơn.

## 5. Mỗi worker có thể có pool riêng

Hệ thống FastAPI production thường chạy nhiều process và nhiều replica. Giả sử có ba replica, mỗi replica có bốn worker process, mỗi worker có một Engine với `pool_size=5`, `max_overflow=5`:

```text
3 replicas × 4 workers × (5 + 5) = 120 connection có thể được mượn
```

Đây là **giới hạn trên theo cấu hình**, không phải 120 connection được tạo ngay hoặc luôn tồn tại. Con số này cũng chưa gồm background jobs, migrations, monitoring và các service khác. PostgreSQL còn dành riêng một số slot cho các role có quyền cao, nên ứng dụng không nhất thiết dùng được toàn bộ `max_connections`.

Vì vậy, thêm worker đôi khi khiến lỗi connection nặng hơn, dù pool của từng worker trông khá nhỏ. Hãy tính pool trên **toàn bộ deployment**. Đồng thời, tránh chia sẻ các pooled connection đã mở giữa các worker process được tạo bằng fork; cần khởi tạo hoặc xử lý lại Engine/pool phù hợp với lifecycle của từng process.

## 6. Connection leak và vòng đời của session

Pool có thể cạn vì session hoặc transaction không được giải phóng đúng lúc. Đoạn code này không bảo đảm cleanup:

```python
async def get_order(order_id: int):
    session = SessionLocal()
    result = await session.execute(select(Order).where(Order.id == order_id))
    return result.scalar_one_or_none()
```

Dùng context manager để vòng đời rõ ràng hơn:

```python
async def get_order(order_id: int):
    async with SessionLocal() as session:
        result = await session.execute(select(Order).where(Order.id == order_id))
        return result.scalar_one_or_none()
```

Session sẽ được đóng khi hàm kết thúc bình thường hoặc có exception. Tuy nhiên, một SQLAlchemy Session **không mặc nhiên giữ một PostgreSQL connection suốt vòng đời của nó**. Session thường lấy connection khi cần và có thể trả lại khi transaction kết thúc. Hãy đo thời gian connection được mượn và phạm vi transaction; số Python Session đang tồn tại không đồng nghĩa với số database connection đang bận.

## 7. `idle in transaction`: không chạy SQL nhưng vẫn giữ tài nguyên

Xét một request như sau:

```python
async with SessionLocal() as session:
    result = await session.execute(select(Order).where(Order.id == order_id))
    await external_service_call()
    return result.scalar_one_or_none()
```

Nếu query đã mở transaction và connection vẫn được mượn trong lúc chờ external service, PostgreSQL có thể hiển thị session là `idle in transaction`. Database không chạy SQL, nhưng connection đó vẫn chưa sẵn sàng cho request khác. Transaction kéo dài còn có thể giữ lock hoặc các phiên bản row cũ và ảnh hưởng tới hoạt động bảo trì.

Nếu external call không bắt buộc nằm trong cùng transaction, hãy hoàn tất thao tác database và lấy ra dữ liệu cần dùng **trước** khi gọi nó:

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

Đừng tách transaction nếu nghiệp vụ thực sự cần nhiều thay đổi diễn ra nguyên tử. FastAPI `yield` dependency có thể quản lý việc đóng session, nhưng với response kéo dài hoặc streaming, cần chú ý cleanup diễn ra **lúc nào**. Khi không còn cần database resource, hãy trả lại sớm.

## 8. Đo phần thời gian đang bị bỏ sót

Hãy tách một API request thành các giai đoạn: xử lý request, chờ lấy connection, thực thi SQL, xử lý kết quả và trả response. Một trace minh họa có thể như sau:

| Giai đoạn | Thời gian |
| :--- | ---: |
| Chờ lấy connection | 8.200 ms |
| Thực thi SQL | 25 ms |
| Xử lý khác | 15 ms |
| Tổng cộng | 8.240 ms |

Chỉ nhìn API latency hoặc chỉ nhìn thời gian SQL đều không chỉ ra hàng đợi này. Hãy theo dõi số connection đang được mượn và còn sẵn trong pool, thời gian chờ lấy connection, số pool timeout và thời gian giữ connection. Ở phía PostgreSQL, xem thêm số session `active`, `idle in transaction`, lock waits và số connection slot còn lại.

Số connection đang được mượn cao có thể bình thường nếu query đang thực hiện công việc hữu ích. Chờ checkout lâu trong khi SQL rất ngắn gợi ý pool quá nhỏ, connection bị giữ quá lâu, hoặc cả hai. Cần đo đạc thay vì dựa vào một ngưỡng duy nhất.

## 9. PostgreSQL cho chúng ta biết gì?

`pg_stat_activity` tổng hợp các server session đang tồn tại:

```sql
SELECT application_name, state, COUNT(*) AS connection_count
FROM pg_stat_activity
WHERE datname = current_database()
GROUP BY application_name, state
ORDER BY connection_count DESC;
```

`active` là đang thực thi; `idle` là đang chờ lệnh mới; `idle in transaction` là transaction còn mở nhưng không có statement nào đang chạy. Một pooled connection chờ được tái sử dụng hoàn toàn có thể ở trạng thái `idle`; riêng điều đó **không chứng minh có leak**.

Để tìm transaction kéo dài và trạng thái chờ:

```sql
SELECT pid, application_name, state,
       NOW() - xact_start AS transaction_age,
       wait_event_type, wait_event
FROM pg_stat_activity
WHERE xact_start IS NOT NULL
ORDER BY transaction_age DESC;
```

Khả năng nhìn thấy thông tin phụ thuộc vào quyền monitoring. Và `pg_stat_activity` không thể cho biết bao nhiêu HTTP request vẫn đang chờ **bên trong pool của ứng dụng**. Cần đo cả hai phía.

## 10. PgBouncer nằm ở đâu?

Khi có nhiều application replica và background worker, ngay cả pool được cấu hình hợp lý ở từng process cũng có thể tạo quá nhiều PostgreSQL server session. **PgBouncer** nằm giữa các client và PostgreSQL, giúp tái sử dụng một số lượng server connection được kiểm soát:

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
    PG --> SP["Server connections được giới hạn"]
    SP --> DB["PostgreSQL"]
```

PgBouncer **không** làm PostgreSQL xử lý được vô hạn công việc đồng thời. Nếu query hoặc transaction vẫn chậm, hàng đợi có thể chỉ chuyển từ ứng dụng sang PgBouncer. Một số deployment giữ app pool nhỏ phía trước PgBouncer; số khác giảm tối đa client-side pooling. Cách kết hợp phù hợp tùy driver, chế độ PgBouncer và workload. Nếu cả hai tầng đều xếp hàng, cần quan sát cả hai.

## 11. Session pooling và transaction pooling

Với **session pooling**, một client giữ một server connection trong suốt client session. Cách này duy trì session-level state, nhưng ít cơ hội chia sẻ server connection hơn giữa các client sống lâu.

Với **transaction pooling**, server connection được cấp cho một transaction và có thể được client khác sử dụng sau đó. Nếu transaction ngắn, chế độ này có thể giảm số server connection cần thiết. Nhưng hai transaction của cùng một client có thể chạy trên hai PostgreSQL backend session khác nhau.

| Tính năng | Session pooling | Transaction pooling |
| :--- | :--- | :--- |
| Transaction thông thường | Hỗ trợ | Hỗ trợ |
| Session-level advisory lock | Hỗ trợ | Không tương thích |
| `LISTEN` | Hỗ trợ | Không tương thích |
| Session state tùy ý qua `SET/RESET` | Hỗ trợ | Không được giữ như mong đợi |
| Temporary table tồn tại qua nhiều transaction | Hỗ trợ | Thường không tương thích |
| Protocol-level named prepared plan | Hỗ trợ | Phụ thuộc cấu hình |

PgBouncer có thể theo dõi protocol-level named prepared statement trong transaction pooling khi `max_prepared_statements` khác 0. Điều này **không có nghĩa** SQL `PREPARE`, mọi statement cache ở driver hay mọi tính năng dựa trên session đều hoạt động như cũ. Hãy kiểm tra đúng phiên bản PgBouncer, driver và cách dùng prepared statement trước khi chuyển chế độ.

## 12. Theo dõi cả PgBouncer

Chạy `SHOW POOLS;` và `SHOW STATS;` trên **admin console của PgBouncer**, không phải gửi chúng như SQL thông thường tới application database. Những trường hữu ích gồm `cl_waiting` (client đang chờ server connection), `sv_active`, `sv_idle` và `maxwait`.

`cl_waiting` tăng có thể do giới hạn pool, transaction dài, SQL chậm hoặc database contention. Tăng PgBouncer pool mà chưa hiểu nguyên nhân có thể chỉ khiến PostgreSQL quá tải. Hãy trace cả đường đi: **HTTP request -> application pool -> PgBouncer -> PostgreSQL**.

## 13. Chọn pool size theo workload

Pool size phù hợp phụ thuộc vào số replica, process, số request thực sự cần database, thời gian giữ connection, thời gian SQL, mức tranh chấp và các client database khác. **HTTP concurrency không bằng database concurrency.** Hai trăm request đang xử lý không nhất thiết cần hai trăm database connection cùng lúc nếu nhiều request đang chờ external service hoặc xử lý trong bộ nhớ.

Giảm thời gian giữ connection có thể giảm pool size cần thiết. Tăng pool size có thể tăng concurrency của database tới khi CPU, I/O, bộ nhớ hoặc lock bắt đầu bị tranh chấp. Query chậm đi sẽ giữ connection lâu hơn, rồi hàng đợi càng dài. Pool tạo **backpressure**, nhưng hàng đợi không giới hạn chỉ biến quá tải thành latency tệ. Hãy đặt thời gian chờ hữu hạn và trả response phù hợp với năng lực cùng SLA của dịch vụ.

## 14. Benchmark hàng đợi, không chỉ benchmark query

Hãy tăng tải từng mức trong điều kiện dataset, SQL, số worker và cache tương đương. Chẳng hạn, so sánh app pool nhỏ (`pool_size=2`, không overflow), pool lớn hơn một chút (`pool_size=5`, `max_overflow=2`) và một cấu hình PgBouncer phù hợp. Đây là **cấu hình để thử nghiệm**, không phải khuyến nghị production.

Đo requests/second; p50/p95/p99 API latency; thời gian lấy và giữ connection; số pool timeout; thời gian SQL; PostgreSQL active sessions, CPU, I/O, lock waits; cùng `cl_waiting` của PgBouncer nếu có. Chạy lặp lại và nhìn tail latency, không chỉ giá trị trung bình.

Hãy thử cả tình huống lỗi: session không được đóng, external call kéo dài trong khi transaction còn mở, và streaming response. Nếu sửa vòng đời connection làm thời gian chờ checkout giảm rõ rệt trong khi thời gian SQL gần như không đổi, đó là bằng chứng mạnh về vị trí nút thắt.

## 15. Trình tự debug trong production

1. Xác định lỗi thuộc pool timeout, connect timeout, statement timeout, lock timeout hay HTTP timeout.
2. Đo thời gian lấy connection tách khỏi thời gian thực thi SQL.
3. Xem `pg_stat_activity`, tuổi transaction và wait events.
4. Đếm tất cả replica, worker, Engine, pool và các client database khác.
5. Tìm session bị leak và database connection bị giữ qua các network call.
6. Nếu connection bị giữ vì SQL chậm, kiểm tra execution plan và lock contention.
7. Sau đó mới điều chỉnh pool limit, application concurrency hoặc PgBouncer và benchmark lại.
8. Sau deployment, theo dõi checkout wait, utilization, error rate và p95/p99 latency.

Trình tự này giúp tránh chữa triệu chứng ở một tầng bằng cách dồn tải sang tầng kế tiếp.

## 16. Những sai lầm thường gặp

- Tăng ngay PostgreSQL `max_connections` mà chưa hiểu vì sao các slot hiện tại bị chiếm.
- Chọn pool size cho một worker rồi quên nhân nó với số process và replica.
- Để session hoặc transaction mở quá lâu, hoặc giữ connection trong lúc gọi external API.
- Gọi mọi PostgreSQL session `idle` là leak, dù pooled connection chờ tái sử dụng cũng có thể `idle`.
- Dùng PgBouncer transaction pooling mà chưa rà soát các tính năng phụ thuộc session.
- Đo thời gian SQL nhưng không đo thời gian chờ checkout.
- Cho rằng async I/O đồng nghĩa PostgreSQL có vô hạn connection hay năng lực xử lý query.

Không có một cấu hình pool tối ưu cho mọi deployment. Hãy điều chỉnh dựa trên workload đo được và vòng đời connection thực tế.

## 17. Phần thời gian bị thiếu trong API latency

Connection pooling giúp tái sử dụng connection tốn kém và kiểm soát concurrency. Nhưng nó cũng có thể tạo một hàng đợi khó nhìn thấy. Pool của SQLAlchemy, PgBouncer và PostgreSQL đều có giới hạn và tín hiệu riêng. Query chạy 20 ms không có nghĩa toàn bộ giai đoạn database chỉ mất 20 ms, nếu request đã chờ connection nhiều giây.

**Pool tốt không phải pool lớn nhất. Đó là pool đủ connection cho workload thực, đồng thời giới hạn concurrency để database vẫn ổn định.** Giữ transaction ngắn, trả tài nguyên kịp thời và đo thời gian chờ trước khi thay đổi giới hạn.

## References và tài liệu đọc thêm

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
