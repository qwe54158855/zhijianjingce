from locust import HttpUser, task, between


class ApiUser(HttpUser):
    wait_time = between(1, 3)

    @task(3)
    def health_check(self):
        self.client.get("/api/v1/health")

    @task(2)
    def list_gallery(self):
        self.client.get("/api/v1/gallery?page=0&size=10")

    @task(1)
    def get_metrics(self):
        self.client.get("/api/v1/metrics/overview")

    @task(1)
    def get_nonexistent_task(self):
        self.client.get("/api/v1/inference/999999", name="/api/v1/inference/[id]")
