from invoke import task

@task
def test(c):
    c.run("poetry run pytest")

@task
def lint(c):
    c.run("poetry run ruff src/ml_data_pipeline")
    c.run("poetry run mypy src/ml_data_pipeline")

@task
def all(c):
    c.run("invoke lint")
    c.run("invoke test")