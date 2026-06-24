import requests
import json
import os

API_URL = "http://127.0.0.1:8000"

def main():

    print("\n" + "=" * 40)
    print("   ZEPHOR RVM - HARDWARE SIMULATOR")
    print("=" * 40)

    while True:

        print("\n🟡 [STANDBY] Waiting for customer...")

        barcode = input(
            "👉 PLUG IN USB SCANNER & SCAN A BOTTLE (or type 'q' to quit): "
        ).strip()

        if barcode.lower() == "q":
            break

        print(
            f"\n📡 [GATE 1] Sending Barcode '{barcode}' to API..."
        )

        try:

            # ----------------------------------
            # STEP 1 : VERIFY BARCODE
            # ----------------------------------

            response_1 = requests.post(
                f"{API_URL}/verify-barcode",
                data={"barcode": barcode}
            )

            if response_1.status_code != 200:

                print(
                    f"❌ [API] REJECTED: "
                    f"{response_1.json()['detail']}"
                )

                continue

            barcode_response = response_1.json()

            print(
                f"✅ [API] {barcode_response['message']}"
            )

            # ----------------------------------
            # STEP 2 : ASK FOR IMAGE PATH
            # ----------------------------------

            image_path = input(
                "\n📂 Enter image path: "
            ).strip()

            if not os.path.exists(image_path):

                print(
                    f"❌ File not found: {image_path}"
                )

                continue

            print(
                f"\n📡 [GATE 2] Uploading image "
                f"for barcode '{barcode}'..."
            )

            # ----------------------------------
            # STEP 3 : SEND IMAGE TO API
            # ----------------------------------

            with open(image_path, "rb") as img_file:

                files = {
                    "file": img_file
                }

                data = {
                    "barcode": barcode
                }

                response_2 = requests.post(
                    f"{API_URL}/verify-bottle",
                    data=data,
                    files=files
                )

            print("\n📊 --- FINAL MACHINE DECISION ---")

            print(
                json.dumps(
                    response_2.json(),
                    indent=2
                )
            )

            print("--------------------------------")

        except requests.exceptions.ConnectionError:

            print(
                "❌ [ERROR] Could not connect.\n"
                "Make sure FastAPI is running:\n"
                "uvicorn main:app --reload"
            )

        except Exception as e:

            print(
                f"❌ Unexpected Error: {str(e)}"
            )


if __name__ == "__main__":
    main()