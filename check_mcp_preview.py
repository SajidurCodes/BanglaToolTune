import asyncio
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from mcp import Client, StdioServerParameters

# Use the existing DataBridge checkout and its stdio test launcher.
databridge_root = Path(r"C:\Github 2\databridge-mcp")
sys.path.insert(0, str(databridge_root))
from databridge.storage import DatasetStore


async def main() -> None:
    with TemporaryDirectory() as temp_dir:
        database_path = Path(temp_dir) / "datasets.sqlite3"
        csv_bytes = (
            b"student_id,class_date,attendance_status\n"
            b"S01,2026-11-15,present\n"
            b"S02,2026-11-15,absent\n"
            b"S03,2026-11-15,present\n"
            b"S04,2026-11-15,late\n"
            b"S05,2026-11-15,present\n"
        )
        dataset = DatasetStore(database_path).add_csv("attendance.csv", csv_bytes)

        server = StdioServerParameters(
            command=sys.executable,
            args=[
                str(databridge_root / "tests" / "mcp_test_server.py"),
                str(database_path),
            ],
        )
        async with asyncio.timeout(45):
            async with Client(server) as client:
                result = await client.call_tool(
                    "preview_dataset",
                    {"dataset_id": dataset.dataset_id, "limit": 4},
                )

        assert result.is_error is False, result.content
        preview = result.structured_content
        assert preview is not None, "Missing structured preview"
        assert preview["columns"] == ["student_id", "class_date", "attendance_status"]
        assert preview["total_rows"] == 5
        assert preview["returned_rows"] == 4
        assert preview["has_more_rows"] is True
        assert preview["rows"] == [
            ["S01", "2026-11-15", "present"],
            ["S02", "2026-11-15", "absent"],
            ["S03", "2026-11-15", "present"],
            ["S04", "2026-11-15", "late"],
        ]
        print("PASS: live MCP preview returned S01 through S04 in order (4 of 5 rows)")


if __name__ == "__main__":
    asyncio.run(main())