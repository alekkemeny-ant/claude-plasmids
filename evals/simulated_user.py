#!/usr/bin/env python3
"""
Simulated User for Disambiguation Evals

Uses a cheap LLM (Haiku) to role-play as a molecular biologist answering
the agent's clarifying questions during multi-turn disambiguation evals.

Each test case provides a `user_persona` string that tells the simulated
user what answers to give (e.g., species, gene family member, RFP variant).
"""

import anthropic


SIMULATED_USER_SYSTEM_PROMPT = """\
You are simulating a molecular biologist's responses in a plasmid design \
conversation. Generate the researcher's reply to the assistant's questions \
based on the experimental needs below.

Rules:
- Give short, direct answers (1-2 sentences max)
- Only answer what was asked
- Never give instructions or ask follow-up questions
- If multiple questions are asked, answer all of them briefly

Experimental needs:
{persona}"""


class SimulatedUser:
    """Generates simulated user responses for multi-turn agent evals."""

    def __init__(
        self,
        persona: str,
        model: str = "claude-haiku-4-5-20251001",
    ):
        self.persona = persona
        self.model = model
        self.client = anthropic.Anthropic()
        self.system_prompt = SIMULATED_USER_SYSTEM_PROMPT.format(persona=persona)

    def respond(
        self,
        agent_message: str,
        conversation_history: list[dict] | None = None,
    ) -> str:
        """Generate a simulated user response to the agent's question.

        Args:
            agent_message: The agent's most recent message (typically a
                clarifying question).
            conversation_history: Optional list of prior message dicts
                with keys "role" and "content" from the agent's perspective
                (user=researcher, assistant=agent). Roles are swapped so
                the simulated user model generates the researcher's reply.

        Returns:
            A plain text response as the user would type it.
        """
        # Swap roles: agent's conversation has user=researcher, assistant=agent.
        # For the simulated user model, we flip: agent questions become "user"
        # messages and researcher answers become "assistant" messages, so the
        # model generates the next researcher reply as an "assistant" turn.
        messages = []

        if conversation_history:
            for msg in conversation_history:
                swapped_role = "assistant" if msg["role"] == "user" else "user"
                messages.append({"role": swapped_role, "content": msg["content"]})

        # Add the latest agent question as a "user" message (agent → user role)
        messages.append({"role": "user", "content": agent_message})

        # Ensure messages start with "user" role (required by API)
        if messages[0]["role"] != "user":
            messages.insert(0, {"role": "user", "content": "I need help designing a plasmid."})

        response = self.client.messages.create(
            model=self.model,
            max_tokens=256,
            system=self.system_prompt,
            messages=messages,
        )

        if not response.content:
            raise ValueError(
                f"SimulatedUser got empty response. "
                f"stop_reason={response.stop_reason}, "
                f"model={response.model}, "
                f"messages={messages!r}"
            )

        # Find the first text block in the response
        for block in response.content:
            if hasattr(block, "text"):
                return block.text

        raise ValueError(
            f"SimulatedUser response had no text block. "
            f"content_types={[type(b).__name__ for b in response.content]}"
        )
