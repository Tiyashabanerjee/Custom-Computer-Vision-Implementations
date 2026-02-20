import os
import sys
import traceback
# setting working dir to the project folder (directory of this script)
os.chdir(os.path.dirname(__file__) or os.getcwd())
sys.path.insert(0, os.path.abspath('.'))
def run():
    print("Running q1...")
    try:
        from q1.main import main as q1main
        q1main()
    except FileNotFoundError as e:
        print("q1 skipped: missing file:", e)
    except Exception:
        print("q1 failed with exception:")
        traceback.print_exc()

    print("Running q2...")
    try:
        from q2.main import main as q2main
        q2main()
    except FileNotFoundError as e:
        print("q2 skipped: missing file:", e)
        print("Hint: put the image at Data\\Question3\\Question2-1.jpg or update the path in q2/main.py")
    except Exception:
        print("q2 failed with exception:")
        traceback.print_exc()

    print("Running q3...")
    try:
        from q3.main import main as q3main
        q3main()
    except FileNotFoundError as e:
        print("q3 skipped: missing file:", e)
    except Exception:
        print("q3 failed with exception:")
        traceback.print_exc()

    print("All done. Check q1/outputs, q2/outputs, q3/outputs.")

if __name__ == '__main__':
    run()