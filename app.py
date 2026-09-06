"""Minimal Streamlit entry point for the FastBI analyst agent."""

import streamlit as st

from agents.analyst_agent import run_agent


def main() -> None:
    """Collect one prompt and display the agent response."""
    st.title("FastBI-Agent")
    user_input = st.text_input("Ask the analyst a question")

    if st.button("Run"):
        if not user_input.strip():
            st.warning("Please enter a question.")
            return

        try:
            with st.spinner("Thinking..."):
                st.write(run_agent(user_input))
        except ValueError as exc:
            st.error(str(exc))


if __name__ == "__main__":
    main()
