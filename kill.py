"""
Global Emergency Kill Switch

Run this script to immediately stop the trading bot:
    python kill.py

What it does:
1. Sends SIGTERM to the running bot process (if found)
2. Sets halt flags in the database for both markets
3. Logs the kill event
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def find_bot_pid() -> int | None:
    """Find the PID of the running main.py process."""
    try:
        result = subprocess.run(
            ["pgrep", "-f", "python.*main.py"],
            capture_output=True, text=True,
        )
        pids = result.stdout.strip().split("\n")
        # Filter out our own PID
        our_pid = os.getpid()
        for pid_str in pids:
            if pid_str.strip():
                pid = int(pid_str.strip())
                if pid != our_pid:
                    return pid
    except Exception:
        pass
    return None


def halt_database() -> None:
    """Set halt flags in the database."""
    try:
        from db.database import Database
        from config.loader import get_config

        config = get_config()
        db = Database(
            db_path=config.database.full_path,
            journal_mode=config.database.journal_mode,
        )

        for market in ("us", "india"):
            db.execute(
                "UPDATE bot_state SET halted = 1, halt_reason = 'EMERGENCY KILL SWITCH' WHERE market = ?",
                (market,),
            )

        db.log_audit("emergency_kill", "kill.py", {"method": "kill_switch"})
        print("[OK] Database halt flags set for both markets")
    except Exception as e:
        print(f"[WARN] Could not update database: {e}")


def main() -> None:
    print("=" * 50)
    print("  EMERGENCY KILL SWITCH")
    print("=" * 50)
    print()

    # 1. Try to kill the process
    pid = find_bot_pid()
    if pid:
        print(f"[KILL] Sending SIGTERM to bot process (PID: {pid})")
        try:
            os.kill(pid, signal.SIGTERM)
            print(f"[OK] SIGTERM sent to PID {pid}")
        except ProcessLookupError:
            print(f"[WARN] Process {pid} already gone")
        except PermissionError:
            print(f"[ERROR] No permission to kill PID {pid}. Try: sudo python kill.py")
    else:
        print("[INFO] No running bot process found")

    # 2. Halt in database
    halt_database()

    print()
    print("[DONE] Kill switch executed. All trading halted.")


if __name__ == "__main__":
    main()
