from mcp.server.fastmcp import FastMCP


mcp = FastMCP("axio-demo-mcp")


@mcp.tool()
def lookup_customer_profile(customer_id: str) -> str:
    """Look up a customer's profile summary."""
    return f"Customer {customer_id} is an active enterprise tenant with standard support."


@mcp.tool()
def get_account_balance(account_id: str) -> str:
    """Retrieve a mock account balance."""
    return f"Account {account_id} has an outstanding balance of 1250 EUR."


if __name__ == "__main__":
    mcp.run()
