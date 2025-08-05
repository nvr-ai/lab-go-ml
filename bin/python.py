from pprint import pprint
import onnxruntime as ort

pprint(f"device: {ort.get_device()}")

pprint(f"providers:")
pprint(sorted(ort.get_all_providers()))

pprint(f"path: {ort.__file__}")