import pynvml
import subprocess
import os
import json
import yaml
import fire
import time

def get_gpu_info():
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()

    gpu_info_list = []
    device_name_indices = {}
    
    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        device_name = pynvml.nvmlDeviceGetName(handle)

        if device_name not in device_name_indices:
            device_name_indices[device_name] = 0
        else:
            device_name_indices[device_name] += 1
        device_index = device_name_indices[device_name]

        device_id = pynvml.nvmlDeviceGetIndex(handle)

        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        total_memory = memory_info.total

        temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)

        gpu_info = {
            'device_id': device_id,
            'device_name': device_name,
            'device_index': device_index,
            'total_memory': total_memory,
            'temperature': temperature
        }
        gpu_info_list.append(gpu_info)

    pynvml.nvmlShutdown()
    return gpu_info_list

def gpu_schedule(config, gpu_info_list):
    resolved_device_ids = []
    for gpu_config in config:
        resolved_ids = []
        for gpu_name, gpu_numbers in gpu_config.items():
            for gpu_number in gpu_numbers:
                found = False
                for gpu_info in gpu_info_list:
                    if gpu_name.lower() in gpu_info['device_name'].lower() and gpu_info['device_index'] == gpu_number:
                        resolved_ids.append(gpu_info['device_id'])
                        found = True
                if not found: print(f'WARNING: Unable to find GPU {gpu_name} {gpu_number}')
        resolved_device_ids.append(resolved_ids)
    return resolved_device_ids

def gpu_temperature_check(device_ids, gpu_info_list, max_temp):
    for device_id in device_ids:
        gpu_info = gpu_info_list[device_id]
        if gpu_info['temperature'] > max_temp:
            print(f"TOO HOT: {gpu_info['device_name']} {gpu_info['device_name']} is {gpu_info['temperature']}C")
            return False
    return True

def start_monitor_gpu(device_id, log_file):
    command = ["nvidia-smi", "dmon", "-s", "petm", "-i", str(device_id)]
    with open(log_file, "w") as f:
        process = subprocess.Popen(command, stdout=f, stderr=subprocess.STDOUT)
        return process

def stop_monitor_gpu(process):
    process.terminate()
    process.wait()

def prepare_tests(device_ids, engine_config, gpu_info_list):
    tests = engine_config.get('tests', [])
    common = engine_config.get('common', {})
    
    prepared_tests = []
    for test in tests:
        test_copy = common.copy()
        print(test_copy, test)
        test_copy.update(test)
        
        if '_min_vram' in test_copy:
            min_vram = test_copy.pop('_min_vram')
            total_vram = sum(gpu_info['total_memory'] for gpu_info in gpu_info_list if gpu_info['device_id'] in device_ids)
            total_vram = total_vram/(1024*1024*1024) # to GB
            if total_vram < min_vram:
                print("Skipping test '{}' due to insufficient VRAM.".format(test_copy.get('_name', 'Unnamed')))
                continue
        
        test_copy['device_ids'] = device_ids
        prepared_tests.append(test_copy)
    
    return prepared_tests

def execute_test(engine, log_file, test):
    print("execute_test", engine, json.dumps(test))
    
    # Convert test object to JSON and write it to a temporary file
    temp_json_file = "temp_test.json"
    with open(temp_json_file, "w") as f:
        json.dump(test, f, indent=4)

    # Convert device_ids to comma-separated list
    device_csv = ','.join(map(str, test['device_ids']))

    # Execute the test command and write output to both screen and log file
    command = [f"{engine}/run.sh", device_csv, temp_json_file]
    with open(log_file, "w") as log_f:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
        for line in process.stdout:
            print(line, end='')
            log_f.write(line)
        process.wait()
    
    # Cleanup temporary JSON file
    os.remove(temp_json_file)

    # Return True if the process exits with a zero status code, indicating success
    return process.returncode == 0

def run_quick_tests():
    gpu_info_list = get_gpu_info()
    for gpu_info in gpu_info_list:
        print("Device ID:", gpu_info['device_id'])
        print("Device Name:", gpu_info['device_name'], "Index:", gpu_info['device_index'])
        print("Total Memory:", gpu_info['total_memory']/(1024*1024), "MB")
        print("Temperature:", gpu_info['temperature'], "C\n")

    # Example configurations
    config = [
        {"RTX 3060": [0]},
        {"P40": [0]},
        {"P100": [0]},
        {"RTX 3060": [0], "P40": [0,1]},
        {"RTX 3060": [0], "P100": [0,1]},
        {"P40": [1]}
    ]
    resolved_device_ids = gpu_schedule(config, gpu_info_list)
    
    for gpus, device_ids in zip(config, resolved_device_ids):
        print('Device spec:', gpus)
        print("Resolved Device IDs:", device_ids)
    
        temp_check_20c = gpu_temperature_check(device_ids, gpu_info_list, 20)
        temp_check_50c = gpu_temperature_check(device_ids, gpu_info_list, 50)
        
        print('Check under 20C:', temp_check_20c, '   Check under 50C:', temp_check_50c)
        print()
        
    # gpu0_monitor = start_monitor_gpu(0, 'test_0.log')
    # time.sleep(5)
    # stop_monitor_gpu(gpu0_monitor)
    
    engine_config = {
        'common': {
            'parameter1': 'value1',
            'parameter2': 'value2'
        },
        'tests': [
            {'_name': 'Test1', 'parameter3': 'value3', '_min_vram': 16},
            {'_name': 'Test2', 'parameter1': 'override'}
        ]
    }
    prepared_tests = prepare_tests([4], engine_config, gpu_info_list)
    for test in prepared_tests:
        execute_test('transformers', 'test.log', test)

def main(config_file:str, min_temp:int = 50):
    for config in yaml.safe_load(open(config_file).read()):
        gpu_info_list = get_gpu_info()
        device_ids_list = gpu_schedule(config['gpus'], gpu_info_list)

        for device_ids in device_ids_list:
            for engine, ecfg in config['engines'].items():
                tests = prepare_tests(device_ids, ecfg, gpu_info_list)
                for test in tests:
                    gpu_info_list = get_gpu_info()
                    while not gpu_temperature_check(device_ids, gpu_info_list, min_temp):
                        print(f"Waiting for all GPUs to reach {min_temp}C...")
                        time.sleep(10)
                        gpu_info_list = get_gpu_info()

                    dmons = []
                    for device in device_ids:
                        dmons.append(start_monitor_gpu(device, f"dmon_{device}.log"))
                    
                    try:
                        execute_test(engine, f"{engine}.log", test)
                    except Exception as e:
                        print('Failed to execute test', test, str(e))
                    
                    for monitor in dmons: stop_monitor_gpu(monitor)       
    
if __name__ == "__main__":
    import fire
    fire.Fire(main)