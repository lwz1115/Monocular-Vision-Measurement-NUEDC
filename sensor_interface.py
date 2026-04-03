# sensor_interface.py
import smbus2

I2C_BUS = 1
INA226_ADDR = 0x40
CURRENT_LSB = 0.0001  # 单位A/bit
POWER_LSB = CURRENT_LSB * 25

def write_calibration(bus, cal_val):
    cal_hi = (cal_val >> 8) & 0xFF
    cal_lo = cal_val & 0xFF
    bus.write_i2c_block_data(INA226_ADDR, 0x05, [cal_hi, cal_lo])

def read_reg16(bus, reg):
    data = bus.read_i2c_block_data(INA226_ADDR, reg, 2)
    raw = (data[0] << 8) | data[1]
    if raw & 0x8000:
        raw -= 0x10000
    return raw

def read_ina226_current_power():
    try:
        bus = smbus2.SMBus(I2C_BUS)
        write_calibration(bus, 0x1400)
        current_raw = read_reg16(bus, 0x04)
        power_raw = read_reg16(bus, 0x03)
        current = current_raw * CURRENT_LSB
        power = power_raw * POWER_LSB
        bus.close()
        return f"{current:.3f} A", f"{power:.3f} W", power
    except Exception as e:
        return "-- A", "-- W", 0.0
