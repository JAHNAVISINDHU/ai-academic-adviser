"""
agent.py
--------
AI Academic Advisor Agent using Claude API with MCP memory tools.
Supports both Claude API and a fallback rule-based agent for testing.

Features:
- Persistent cross-session memory via MCP server
- Semantic retrieval of relevant past conversations
- Academic milestone and preference tracking
- Interactive CLI interface
"""

import os
import json
import logging
import time
import httpx
from datetime import datetime
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8000")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# ─── MCP Client ───────────────────────────────────────────────────────────────

class MCPClient:
    """HTTP client for the MCP memory server."""

    def __init__(self, base_url: str = MCP_SERVER_URL):
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(timeout=30.0)

    def health(self) -> bool:
        try:
            r = self.client.get(f"{self.base_url}/health")
            return r.status_code == 200
        except Exception:
            return False

    def memory_write(self, memory_type: str, data: dict) -> dict:
        r = self.client.post(
            f"{self.base_url}/invoke/memory_write",
            json={"memory_type": memory_type, "data": data},
        )
        r.raise_for_status()
        return r.json()

    def memory_read(self, user_id: str, query_type: str, params: dict = None) -> dict:
        r = self.client.post(
            f"{self.base_url}/invoke/memory_read",
            json={"user_id": user_id, "query_type": query_type, "params": params or {}},
        )
        r.raise_for_status()
        return r.json()

    def memory_retrieve_by_context(self, user_id: str, query_text: str, top_k: int = 3) -> dict:
        r = self.client.post(
            f"{self.base_url}/invoke/memory_retrieve_by_context",
            json={"user_id": user_id, "query_text": query_text, "top_k": top_k},
        )
        r.raise_for_status()
        return r.json()


# ─── Claude-powered Agent ─────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert AI Academic Advisor with long-term memory capabilities.
You help students plan their academic journey, track their progress, set goals, and provide 
personalized guidance based on their history, preferences, and milestones.

You have access to three memory tools:

1. **memory_write** - Save conversation turns, user preferences, or milestones
2. **memory_read** - Retrieve structured data (last N turns, preferences, milestones)  
3. **memory_retrieve_by_context** - Semantic search of past conversations

MEMORY STRATEGY:
- At the start of each response, ALWAYS call memory_retrieve_by_context to find relevant past context
- After generating your response, ALWAYS call memory_write to save the conversation turn
- When the user mentions preferences (favorite subjects, goals, study style), save them with memory_write (preference type)
- When the user achieves or mentions goals, save them as milestones

TONE: Encouraging, knowledgeable, and personalized. Reference past conversations naturally.
Keep responses concise but helpful (2-4 paragraphs max).
"""

TOOLS = [
    {
        "name": "memory_write",
        "description": "Save a memory (conversation turn, user preference, or milestone) to persistent storage",
        "input_schema": {
            "type": "object",
            "properties": {
                "memory_type": {"type": "string", "enum": ["conversation", "preference", "milestone"]},
                "data": {"type": "object"},
            },
            "required": ["memory_type", "data"],
        },
    },
    {
        "name": "memory_read",
        "description": "Read structured data from storage (last N conversation turns, preferences, or milestones)",
        "input_schema": {
            "type": "object",
            "properties": {
                "user_id": {"type": "string"},
                "query_type": {"type": "string", "enum": ["last_n_turns", "all_preferences", "all_milestones"]},
                "params": {"type": "object"},
            },
            "required": ["user_id", "query_type"],
        },
    },
    {
        "name": "memory_retrieve_by_context",
        "description": "Semantic search: find past memories most relevant to a query",
        "input_schema": {
            "type": "object",
            "properties": {
                "user_id": {"type": "string"},
                "query_text": {"type": "string"},
                "top_k": {"type": "integer"},
            },
            "required": ["user_id", "query_text"],
        },
    },
]


class AcademicAdvisorAgent:
    """AI Academic Advisor with persistent memory."""

    def __init__(self, user_id: str):
        self.user_id = user_id
        self.mcp = MCPClient()
        self.turn_counter = self._get_next_turn()
        self.use_claude = bool(ANTHROPIC_API_KEY)

        if self.use_claude:
            import anthropic
            self.claude = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            logger.info("Using Claude API for agent responses")
        else:
            logger.info("No ANTHROPIC_API_KEY found. Using rule-based fallback agent.")

    def _get_next_turn(self) -> int:
        """Get the next turn ID by checking existing conversations."""
        try:
            result = self.mcp.memory_read(self.user_id, "last_n_turns", {"n": 1})
            if result.get("results"):
                return result["results"][-1]["turn_id"] + 1
        except Exception:
            pass
        return 1

    def _execute_tool(self, tool_name: str, tool_input: dict) -> str:
        """Execute an MCP tool and return the result as a string."""
        try:
            # Inject user_id where needed
            if "user_id" not in tool_input:
                tool_input["user_id"] = self.user_id
            if tool_name == "memory_write":
                if "data" in tool_input and "user_id" not in tool_input["data"]:
                    tool_input["data"]["user_id"] = self.user_id
                result = self.mcp.memory_write(tool_input["memory_type"], tool_input["data"])
            elif tool_name == "memory_read":
                result = self.mcp.memory_read(
                    tool_input["user_id"],
                    tool_input["query_type"],
                    tool_input.get("params", {}),
                )
            elif tool_name == "memory_retrieve_by_context":
                result = self.mcp.memory_retrieve_by_context(
                    tool_input["user_id"],
                    tool_input["query_text"],
                    tool_input.get("top_k", 3),
                )
            else:
                result = {"error": f"Unknown tool: {tool_name}"}
            return json.dumps(result)
        except Exception as e:
            logger.error(f"Tool {tool_name} failed: {e}")
            return json.dumps({"error": str(e)})

    def chat_with_claude(self, user_message: str) -> str:
        """Run the agentic loop with Claude API."""
        messages = [{"role": "user", "content": user_message}]

        while True:
            response = self.claude.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2048,
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                messages=messages,
            )

            # Collect text from response
            text_output = ""
            tool_calls = []

            for block in response.content:
                if block.type == "text":
                    text_output += block.text
                elif block.type == "tool_use":
                    tool_calls.append(block)

            # If no tool calls, we're done
            if not tool_calls or response.stop_reason == "end_turn":
                # Save the assistant response to memory
                if text_output.strip():
                    self._save_turn("assistant", text_output)
                return text_output

            # Execute tool calls
            messages.append({"role": "assistant", "content": response.content})
            tool_results = []
            for tc in tool_calls:
                result = self._execute_tool(tc.name, tc.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tc.id,
                    "content": result,
                })
            messages.append({"role": "user", "content": tool_results})

    def chat_fallback(self, user_message: str) -> str:
        """Rule-based fallback agent when Claude API is not available."""
        # Retrieve relevant context
        try:
            context_result = self.mcp.memory_retrieve_by_context(
                self.user_id, user_message, top_k=3
            )
            context = context_result.get("results", [])
        except Exception:
            context = []

        # Build a personalized response based on context
        context_str = ""
        if context:
            top = context[0]
            context_str = f"\n\n(Relevant context from our past conversations: \"{top['content']}\")"

        # Generate a relevant response
        lower_msg = user_message.lower()
        if any(w in lower_msg for w in ["major", "study", "degree", "field"]):
            response = (
                f"Great question about your academic path! Based on what you've shared, "
                f"I'd recommend exploring programs that align with your interests. "
                f"Have you considered speaking with faculty advisors in your area of interest? "
                f"They can provide invaluable insight into course requirements and research opportunities."
                f"{context_str}"
            )
        elif any(w in lower_msg for w in ["course", "class", "enroll", "schedule"]):
            response = (
                f"Course planning is crucial for staying on track. I recommend mapping out "
                f"your required courses first, then filling in electives that complement your major. "
                f"Don't forget to check prerequisites and plan for challenging semesters in advance!"
                f"{context_str}"
            )
        elif any(w in lower_msg for w in ["grade", "gpa", "exam", "test", "homework"]):
            response = (
                f"Academic performance matters, but remember it's about learning, not just grades. "
                f"For exams: start studying 2 weeks early, use active recall techniques, and form study groups. "
                f"Office hours are your secret weapon — professors appreciate engaged students!"
                f"{context_str}"
            )
        elif any(w in lower_msg for w in ["career", "job", "internship", "work"]):
            response = (
                f"Career development starts early! I'd recommend: attending career fairs, "
                f"building your LinkedIn profile, seeking internships from sophomore year, "
                f"and connecting with alumni in your field. Your university's career center "
                f"is an excellent (and often underused) resource."
                f"{context_str}"
            )
        elif any(w in lower_msg for w in ["stress", "overwhelm", "anxiety", "tired", "burnout"]):
            response = (
                f"I hear you — academic life can be overwhelming. Remember: your wellbeing "
                f"comes first. Take breaks, maintain social connections, exercise regularly, "
                f"and don't hesitate to use campus mental health resources. "
                f"Academic success is a marathon, not a sprint."
                f"{context_str}"
            )
        else:
            response = (
                f"Thank you for sharing that with me. As your academic advisor, I'm here to help "
                f"you navigate your educational journey. Could you tell me more about your specific "
                f"goals or challenges? The more I know about you, the better I can personalize my guidance."
                f"{context_str}"
            )

        return response

    def _save_turn(self, role: str, content: str):
        """Save a conversation turn to memory."""
        try:
            self.mcp.memory_write("conversation", {
                "user_id": self.user_id,
                "turn_id": self.turn_counter,
                "role": role,
                "content": content,
                "timestamp": datetime.utcnow().isoformat(),
            })
            self.turn_counter += 1
        except Exception as e:
            logger.error(f"Failed to save turn: {e}")

    def chat(self, user_message: str) -> str:
        """Main chat entry point."""
        # Save user message to memory
        self._save_turn("user", user_message)

        if self.use_claude:
            response = self.chat_with_claude(user_message)
        else:
            response = self.chat_fallback(user_message)
            # Save assistant response
            self._save_turn("assistant", response)

        return response


# ─── Main CLI Loop ────────────────────────────────────────────────────────────

def wait_for_mcp(mcp: MCPClient, max_wait: int = 60):
    """Wait for the MCP server to become healthy."""
    logger.info(f"Waiting for MCP server at {MCP_SERVER_URL}...")
    for i in range(max_wait):
        if mcp.health():
            logger.info("MCP server is healthy!")
            return True
        time.sleep(1)
        if i % 10 == 9:
            logger.info(f"Still waiting... ({i+1}s)")
    return False


def main():
    print("\n" + "="*60)
    print("  🎓 AI Academic Advisor - Powered by MCP Memory")
    print("="*60)

    mcp = MCPClient()
    if not wait_for_mcp(mcp, max_wait=60):
        print("❌ Could not connect to MCP server. Please ensure it is running.")
        return

    print("\nConnected to MCP memory server ✓")

    # Get or create user ID
    print("\nEnter your student ID (or press Enter for 'student_001'):")
    user_id = input("  > ").strip() or "student_001"

    print(f"\nWelcome, {user_id}! I'm your AI Academic Advisor.")
    print("I have persistent memory — I'll remember our conversations across sessions.")
    print("Type 'quit' to exit, 'history' to see past conversations, or 'milestones' to view your goals.\n")

    agent = AcademicAdvisorAgent(user_id=user_id)

    # Show past context if available
    try:
        history = mcp.memory_read(user_id, "last_n_turns", {"n": 3})
        if history.get("results"):
            print(f"📚 I remember our previous conversations. Here's what we last discussed:")
            for turn in history["results"][-2:]:
                role_emoji = "👤" if turn["role"] == "user" else "🎓"
                print(f"  {role_emoji} {turn['content'][:80]}{'...' if len(turn['content']) > 80 else ''}")
            print()
    except Exception:
        pass

    # Main conversation loop
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye! Your progress has been saved. 👋")
            break

        if not user_input:
            continue

        if user_input.lower() == "quit":
            print("\nGoodbye! Your progress has been saved. Best of luck with your studies! 👋")
            break

        elif user_input.lower() == "history":
            try:
                result = mcp.memory_read(user_id, "last_n_turns", {"n": 10})
                print("\n📜 Recent conversation history:")
                for turn in result.get("results", []):
                    role_emoji = "👤" if turn["role"] == "user" else "🎓"
                    print(f"  {role_emoji} [{turn['timestamp'][:10]}] {turn['content'][:100]}")
                print()
            except Exception as e:
                print(f"  Error retrieving history: {e}\n")
            continue

        elif user_input.lower() == "milestones":
            try:
                result = mcp.memory_read(user_id, "all_milestones", {})
                milestones = result.get("results", [])
                if milestones:
                    print("\n🏆 Your Academic Milestones:")
                    for ms in milestones:
                        emoji = "✅" if ms["status"] == "completed" else "🔄" if ms["status"] == "in-progress" else "📋"
                        print(f"  {emoji} [{ms['status']}] {ms['description']}")
                else:
                    print("\n  No milestones recorded yet. Tell me about your academic goals!")
                print()
            except Exception as e:
                print(f"  Error retrieving milestones: {e}\n")
            continue

        # Generate and display response
        print("\n🎓 Advisor: ", end="", flush=True)
        response = agent.chat(user_input)
        print(response)
        print()


if __name__ == "__main__":
    main()
