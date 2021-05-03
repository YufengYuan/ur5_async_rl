import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import animation
import time


from senseact.communicator import Communicator
from senseact.sharedbuffer import SharedBuffer

class MonitorCommunicator(Communicator):

    def __init__(self, target_type='reacher', width=160, height=90, radius=7):
        mpl.rcParams['toolbar'] = 'None'
        plt.ion()
        self.fig = plt.figure()
        plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
        self.fig.canvas.toolbar_visible = False
        self.ax = plt.axes(xlim=(0, width), ylim=(0, height))
        self.target = plt.Circle((0, 0), radius, color='red')
        self.ax.add_patch(self.target)
        plt.axis('off')
        self.radius = radius
        self.width = width
        self.height = height
        self.target_type=target_type
        actuator_args = {
            'array_len': 1,
            'array_type': 'd',
            'np_array_type': 'd',
        }
        super(MonitorCommunicator, self).__init__(
            use_sensor=False,
            use_actuator=True,
            sensor_args={},
            actuator_args=actuator_args
        )
        self.reset()
        figManager = plt.get_current_fig_manager()
        figManager.full_screen_toggle()

    def reset(self):
        if self.target_type == 'static':
            self.target.set_center((self.width / 2, self.height / 2))
            self.velocity_x, self.velocity_y = 0, 0
        if self.target_type == 'reacher':
            x, y = np.random.random(2)
            self.target.set_center(
                (self.radius + x * (self.width - 2 * self.radius),
                 self.radius + y * (self.height - 2 * self.radius))
            )
            self.velocity_x, self.velocity_y = 0, 0
        elif self.target_type == 'tracker':
            #x, y = np.random.random(2)
            #self.target.set_center(
            #    (self.radius + x * (self.width - 2 * self.radius),
            #     self.radius + y * (self.height - 2 * self.radius))
            #)
            self.target.set_center((self.width / 2, self.height / 2))
            self.velocity_x, self.velocity_y = np.random.random(2) - 0.5
            velocity = np.sqrt(self.velocity_x ** 2 + self.velocity_y ** 2)
            self.velocity_x /= velocity
            self.velocity_y /= velocity

    def run(self):
        super(MonitorCommunicator, self).run()

    def _sensor_handler(self):
        raise NotImplementedError('fsf')

    def _actuator_handler(self):
        if self.actuator_buffer.updated():
            self.actuator_buffer.read_update()
            self.reset()
        x, y = self.target.get_center()
        if x + self.velocity_x + self.radius > self.width or \
           x + self.velocity_x - self.radius < 0:
            self.velocity_x = -self.velocity_x
        if y + self.velocity_y + self.radius > self.height or \
           y + self.velocity_y - self.radius < 0:
            self.velocity_y = -self.velocity_y
        self.target.set_center((x + self.velocity_x, y + self.velocity_y))
        time.sleep(0.032)
        #self.fig.canvas.toolbar.pack_forget()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


if __name__ == '__main__':
   p = MonitorCommunicator()
   p.start()
   while True:
       time.sleep(5)
       p.actuator_buffer.write(0)





