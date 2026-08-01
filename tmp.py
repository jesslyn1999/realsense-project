from scservo_sdk import COMM_SUCCESS, PortHandler, sms_sts

DEVICE = "/dev/ttyACM0"
BAUDRATE = 1_000_000


def main() -> None:
    port = PortHandler(DEVICE)

    try:
        if not port.openPort():
            raise RuntimeError(f"Could not open {DEVICE}")

        if not port.setBaudRate(BAUDRATE):
            raise RuntimeError(f"Could not set {BAUDRATE:,} baud")

        bus = sms_sts(port)
        found = []

        print(f"Scanning IDs 0–253 on {DEVICE} at {BAUDRATE:,} baud...")

        for servo_id in range(254):
            model, comm_result, error = bus.ping(servo_id)

            if comm_result == COMM_SUCCESS:
                found.append(
                    {
                        "id": servo_id,
                        "model": model,
                        "error": error,
                    }
                )
                print(
                    f"Found servo: ID={servo_id}, "
                    f"model={model}, error={error}"
                )

        if found:
            print(f"\nFound {len(found)} servo(s):")
            for servo in found:
                print(
                    f"ID={servo['id']}, "
                    f"model={servo['model']}, "
                    f"error={servo['error']}"
                )
        else:
            print("\nNo servos found.")

    finally:
        port.closePort()
        print("Serial port closed.")


if __name__ == "__main__":
    main()