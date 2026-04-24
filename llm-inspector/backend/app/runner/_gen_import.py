import pathlib, json, sys
sys.path.insert(0, "backend")

code = """""" + open('backend/app/runner/cache_strategy.py').read()[:50]
print(code[:30])

# Actually just write import_dataset.py directly
content = open('backend/app/handlers/v15_handlers.py').read()[:200]
gen2 = pathlib.Path('backend/app/runner/import_dataset.py')
gen2.write_text('# test
print(1)
')
print('done')
