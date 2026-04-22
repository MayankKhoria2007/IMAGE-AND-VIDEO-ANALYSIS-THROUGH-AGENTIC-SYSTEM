import time
import uuid


class CommunicationAgent:
    def __init__(self):
        self.log = []
        self.agents = {}

 
    # Register Agents
 
    def register_agents(self, agents_dict):
        """
        agents_dict = {
            "vision": vision_agent,
            "context": context_agent,
            ...
        }
        """
        self.agents = agents_dict

 
    # Create Message
 
    def create_message(self, sender, receiver, data, msg_type="data"):
        message = {
            "id": str(uuid.uuid4()),
            "sender": sender,
            "receiver": receiver,
            "type": msg_type,
            "data": data,
            "timestamp": time.time()
        }
        return message

 
    # Validate Message
 
    def validate_message(self, message):
        required_keys = ["sender", "receiver", "data"]

        for key in required_keys:
            if key not in message:
                return False, f"Missing key: {key}"

        if message["receiver"] not in self.agents:
            return False, f"Unknown receiver: {message['receiver']}"

        return True, "Valid"

 
    # Send Message (log + validate)
 
    def send(self, sender, receiver, data, msg_type="data"):
        message = self.create_message(sender, receiver, data, msg_type)

        valid, error = self.validate_message(message)
        if not valid:
            raise ValueError(f"Invalid message: {error}")

        self.log.append(message)
        return message

 
    # Route Message
 
    def route(self, message):
        receiver = message["receiver"]
        agent = self.agents.get(receiver)

        try:
            response = agent.run(message["data"])

            # log response
            self.log.append({
                "reply_to": message["id"],
                "sender": receiver,
                "data": response,
                "timestamp": time.time()
            })

            return response

        except Exception as e:
            error_msg = {
                "error": str(e),
                "agent": receiver,
                "message_id": message["id"]
            }
            self.log.append(error_msg)
            raise RuntimeError(f"Agent {receiver} failed: {str(e)}")

 
    # Send + Route (shortcut)
 
    def dispatch(self, sender, receiver, data, msg_type="data"):
        message = self.send(sender, receiver, data, msg_type)
        return self.route(message)

 
    # Broadcast Message
 
    def broadcast(self, sender, receivers, data, msg_type="broadcast"):
        responses = {}

        for receiver in receivers:
            try:
                responses[receiver] = self.dispatch(sender, receiver, data, msg_type)
            except Exception as e:
                responses[receiver] = f"Error: {str(e)}"

        return responses

 
    # Retry Mechanism
 
    def retry(self, message, retries=2, delay=1):
        for attempt in range(retries):
            try:
                return self.route(message)
            except Exception:
                time.sleep(delay)

        raise RuntimeError("Max retries exceeded")

 
    # Get Logs
 
    def get_logs(self):
        return self.log

 
    # Pretty Print Logs
 
    def print_logs(self):
        for entry in self.log:
            print(entry)

 
    # Clear Logs
 
    def clear_logs(self):
        self.log = []