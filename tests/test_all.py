import os


test_dir = 'tests'
passed = []
all_files = os.listdir(test_dir)

for i, file in enumerate(os.listdir(test_dir)):
    if not file.endswith('.py') or file == 'test_all.py':
        continue
    file = os.path.join(test_dir, file)
    print('Test ', file)
    ret = os.system(f'python {file}')
    if ret != 0:
        print(f'File {file} test error!')
        print('passed files', all_files[:i])
        print('rest files', all_files[i+1:])
        break
else:
    print('Test all OK!')
