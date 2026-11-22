# migrations

Alembic migrations for the PostgreSQL schema in `app/db/models.py`.

```bash
make db-up      # local Postgres from compose.yaml
make migrate    # alembic upgrade head against DATABASE_URL
```

`DATABASE_URL` defaults to the compose database
(`postgresql+psycopg://eval:eval_local_only@127.0.0.1:5432/agent_eval`); the URL is
never read from `alembic.ini`.

To add a revision, change the models, then run
`alembic revision --autogenerate -m "..."` against a migrated database and review
the result. `tests/postgres/test_migrations.py` fails if models and migrations drift.
