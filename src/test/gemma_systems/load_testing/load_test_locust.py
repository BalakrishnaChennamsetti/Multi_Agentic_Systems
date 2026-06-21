from locust import HttpUser, task


class ChatUser(HttpUser):

    @task
    def chat(self):
        self.client.post("/process", json={"text": "hello"})
