import socket
import struct
from threading import Thread
from time import sleep, perf_counter
import numpy as np


class ForceSensor:
    def __init__(self, ip="192.168.1.1"):
        self.sensor = Sensor(ip)
        self.COUNTS_PER_FORCE = 1000000  # obtain from http://192.168.1.1/config.htm
        self.COUNTS_PER_TORQUE = 1000000  # obtain from http://192.168.1.1/config.htm
        try:
            self.sensor.tare()  # zero in air
            self.start_time = perf_counter()
        except Exception as e:
            assert False, f"[ForceSensor] Failed to connect to force sensor: {str(e)}"
        print("Force sensor initialized")
        self.sensor.startStreaming()

    def reset(self):
        self.sensor.stopStreaming()
        self.sensor.tare()
        self.start_time = perf_counter()
        self.sensor.startStreaming()

    def close(self):
        self.sensor.stopStreaming()

    def get_data(self):
        try:
            fx, fy, fz, tx, ty, tz = self.sensor.measurement()
            force_data = np.array([fx, fy, fz, tx, ty, tz])
            force_data[:3] = force_data[:3] / self.COUNTS_PER_FORCE
            force_data[3:] = force_data[3:] / self.COUNTS_PER_TORQUE
            return force_data
        except Exception as e:
            # return previous data
            print(f"[ForceSensor] Error reading force sensor: {str(e)}")
            return np.zeros(6)


class Sensor:
    """The class interface for an ATI Force/Torque sensor.

    This class contains all the functions necessary to communicate
    with an ATI Force/Torque sensor with a Net F/T interface
    using RDT.
    """

    def __init__(self, ip):
        """Start the sensor interface

        This function initializes the class and opens the socket for the
        sensor.

        Args:
                ip (str): The IP address of the Net F/T box.
        """
        self.ip = ip
        self.port = 49152
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.connect((ip, self.port))
        self.mean = [0] * 6
        self.stream = False

    def send(self, command, count=0):
        """Send a given command to the Net F/T box with specified sample count.

        This function sends the given RDT command to the Net F/T box, along with
        the specified sample count, if needed.

        Args:
                command (int): The RDT command.
                count (int, optional): The sample count to send. Defaults to 0.
        """
        header = 0x1234
        message = struct.pack("!HHI", header, command, count)
        self.sock.send(message)

    def receive(self):
        """Receives and unpacks a response from the Net F/T box.

        This function receives and unpacks an RDT response from the Net F/T
        box and saves it to the data class attribute.

        Returns:
                list of float: The force and torque values received. The first three
                        values are the forces recorded, and the last three are the measured
                        torques.
        """
        while True:
            try:
                rawdata = self.sock.recv(1024)
                data = struct.unpack("!IIIiiiiii", rawdata)[3:]
                self.data = [data[i] - self.mean[i] for i in range(6)]
                break
            except Exception as e:
                sleep(0.01)
        return self.data

    def tare(self, n=10):
        """Tare the sensor.

        This function takes a given number of readings from the sensor
        and averages them. This mean is then stored and subtracted from
        all future measurements.

        Args:
                n (int, optional): The number of samples to use in the mean.
                        Defaults to 10.

        Returns:
                list of float: The mean values calculated.
        """
        self.mean = [0] * 6
        self.getMeasurements(n=n)
        mean = [0] * 6
        for i in range(n):
            self.receive()
            for i in range(6):
                mean[i] += self.measurement()[i] / float(n)
        self.mean = mean
        return mean

    def zero(self):
        """Remove the mean found with `tare` to start receiving raw sensor values."""
        self.mean = [0] * 6

    def receiveHandler(self):
        """A handler to receive and store data."""
        while self.stream:
            self.receive()

    def getMeasurement(self):
        """Get a single measurement from the sensor

        Request a single measurement from the sensor and return it. If
        The sensor is currently streaming, started by running `startStreaming`,
        then this function will simply return the most recently returned value.

        Returns:
                list of float: The force and torque values received. The first three
                        values are the forces recorded, and the last three are the measured
                        torques.
        """
        self.getMeasurements(1)
        self.receive()
        return self.data

    def measurement(self):
        """Get the most recent force/torque measurement

        Returns:
                list of float: The force and torque values received. The first three
                        values are the forces recorded, and the last three are the measured
                        torques.
        """
        return self.data

    def getForce(self):
        """Get a single force measurement from the sensor

        Request a single measurement from the sensor and return it.

        Returns:
                list of float: The force values received.
        """
        return self.getMeasurement()[:3]

    def force(self):
        """Get the most recent force measurement

        Returns:
                list of float: The force values received.
        """
        return self.measurement()[:3]

    def getTorque(self):
        """Get a single torque measurement from the sensor

        Request a single measurement from the sensor and return it.

        Returns:
                list of float: The torque values received.
        """
        return self.getMeasurement()[3:]

    def torque(self):
        """Get the most recent torque measurement

        Returns:
                list of float: The torque values received.
        """
        return self.measurement()[3:]

    def startStreaming(self, handler=True):
        """Start streaming data continuously

        This function commands the Net F/T box to start sending data continuously.
        By default this also starts a new thread with a handler to save all data
        points coming in. These data points can still be accessed with `measurement`,
        `force`, and `torque`. This handler can also be disabled and measurements
        can be received manually using the `receive` function.

        Args:
                handler (bool, optional): If True start the handler which saves data to be
                        used with `measurement`, `force`, and `torque`. If False the
                        measurements must be received manually. Defaults to True.
        """
        self.getMeasurements(0)
        print("Started streaming data")
        if handler:
            self.stream = True
            self.thread = Thread(target=self.receiveHandler)
            self.thread.daemon = True
            self.thread.start()

    def getMeasurements(self, n):
        """Request a given number of samples from the sensor

        This function requests a given number of samples from the sensor. These
        measurements must be received manually using the `receive` function.

        Args:
                n (int): The number of samples to request.
        """
        self.send(2, count=n)

    def stopStreaming(self):
        """Stop streaming data continuously

        This function stops the sensor from streaming continuously as started using
        `startStreaming`.
        """
        self.stream = False
        sleep(0.1)
        self.send(0)


if __name__ == "__main__":
    sensor = ForceSensor()
    count = 0
    while True:
        print(f"{count}: {sensor.get_data()}")
        count += 1
        sleep(0.1)
        if count % 20 == 0:
            sensor.reset()
            print("Reset")
