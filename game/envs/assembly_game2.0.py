import pygame
import time
import  random
class ComputerAssemblyGame:
    def __init__(self):
        pygame.init()
        self.screen_width = 1500
        self.screen_height = 1000
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        self.grid_size = 100  # 设置网格大小为100像素
        pygame.display.set_caption("电脑装配游戏")

        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.clock = pygame.time.Clock()
        self.case_x, self.case_y = 0, 200  # 机箱的位置
        self.case_width, self.case_height = 400, 400  # 机箱的大小
        self.messages = []  # 存储要显示的消息及其时间戳
        self.background_image = pygame.transform.scale(pygame.image.load('./image/background.png'), (self.screen_width, self.screen_height))
        self.case_image = pygame.transform.scale(pygame.image.load('./image/case.png'),(self.case_width,self.case_height))
        self.last_keys = pygame.key.get_pressed()

        # 初始化手的位置和图像
        self.initialize_hands()

        # 初始化零件
        self.initialize_components()
        self.pickup_threshold = 80

        self.running = True
        # 初始化组件状态
        self.states = {
            'motherboard': {'position': (0, 0), 'state': 1},
            'CPU': {'position': (0, 0), 'state': 1},
            'cooler': {'position': (0, 0), 'state': 1},
            'GPU': {'position': (0, 0), 'state': 1},
            'memory_card': {'position': (0, 0), 'state': 1},
            'power_supply': {'position': (0, 0), 'state': 1},
            'hard_disk': {'position': (0, 0), 'state': 1},
            'cover': {'position': (0, 0), 'state': 1}
        }
        self.log_file_path = './data/state_log.txt'
        self.write_log_header()
        self.action_log_file_path = './data/action_log.txt'
        self.create_action_log_file()
        # 初始化树结构
        self.initialize_tree()

    def initialize_components(self):
        self.components = []
        used_positions = set()  # Track used grid positions

        component_names = ['motherboard', 'CPU', 'cooler', 'GPU', 'memory_card', 'power_supply', 'hard_disk', 'cover']
        image_paths = ['./image/motherboard.png', './image/cpu.png', './image/cooler.png', './image/gpu.png',
                       './image/memory_card.png', './image/power_supply.png', './image/hard_disk.png',
                       './image/cover.png']

        for name, image_path in zip(component_names, image_paths):
            while True:
                grid_x = random.randint(4, (self.screen_width - self.grid_size) // self.grid_size)
                grid_y = random.randint(0, (self.screen_height - self.grid_size) // self.grid_size)
                x = grid_x * self.grid_size
                y = grid_y * self.grid_size
                if (grid_x, grid_y) not in used_positions:
                    used_positions.add((grid_x, grid_y))
                    break
            self.components.append(Component(x, y, self.grid_size, self.grid_size, name, image_path))
        self.placement_positions = {
            'motherboard': (70, 250),
            'CPU': (170, 250),
            'cooler': (170, 350),
            'GPU': (270, 250),
            'memory_card': (70, 350),
            'power_supply': (270, 350),
            'hard_disk': (270, 450),
            'cover': (0, 200)
            # ... 添加其他零件的位置 ...
        }

    def move_hand1_to_component(self, component_id):
        # Identify the target component's position
        if component_id < 1 or component_id > len(self.components):
            print("Invalid component ID")
            return

        target_component = self.components[component_id - 1]  # Assuming component_id starts from 1
        target_x, target_y = target_component.x, target_component.y

        # Determine the direction to move in the x and y axis
        dx = target_x - self.hand_x
        dy = target_y - self.hand_y

        # Movement speed (grid sizes per second)
        speed = 3 * self.grid_size
        move_time = 1 / 60  # Assuming the game updates at 60 FPS

        # Calculate the number of frames to move one grid size
        frames_per_grid = 60 / speed

        while dx != 0 or dy != 0:
            if dx != 0:
                step_x = min(abs(dx), self.grid_size) * (dx // abs(dx))
                self.hand_x += step_x
                dx -= step_x
                # Simulate time passing for one grid movement
                time.sleep(frames_per_grid * move_time)
            elif dy != 0:
                step_y = min(abs(dy), self.grid_size) * (dy // abs(dy))
                self.hand_y += step_y
                dy -= step_y
                # Simulate time passing for one grid movement
                time.sleep(frames_per_grid * move_time)

            self.update()  # Update game state and render
            self.render()

    def update(self):
        current_keys = pygame.key.get_pressed()

        move_distance = self.grid_size  # Move by one grid cell

        # Check for state change from not pressed to pressed
        if current_keys[pygame.K_a] and not self.last_keys[pygame.K_a]:
            self.hand_x = max(self.hand_x - move_distance, 0)
        if current_keys[pygame.K_d] and not self.last_keys[pygame.K_d]:
            self.hand_x = min(self.hand_x + move_distance, self.screen_width - self.grid_size)
        if current_keys[pygame.K_w] and not self.last_keys[pygame.K_w]:
            self.hand_y = max(self.hand_y - move_distance, 0)
        if current_keys[pygame.K_s] and not self.last_keys[pygame.K_s]:
            self.hand_y = min(self.hand_y + move_distance, self.screen_height - self.grid_size)

        if current_keys[pygame.K_LEFT] and not self.last_keys[pygame.K_LEFT]:
            self.second_hand_x = max(self.second_hand_x - move_distance, 0)
        if current_keys[pygame.K_RIGHT] and not self.last_keys[pygame.K_RIGHT]:
            self.second_hand_x = min(self.second_hand_x + move_distance, self.screen_width - self.grid_size)
        if current_keys[pygame.K_UP] and not self.last_keys[pygame.K_UP]:
            self.second_hand_y = max(self.second_hand_y - move_distance, 0)
        if current_keys[pygame.K_DOWN] and not self.last_keys[pygame.K_DOWN]:
            self.second_hand_y = min(self.second_hand_y + move_distance, self.screen_height - self.grid_size)

        # Update hand rects for collision detection
        self.hand1_rect.topleft = (self.hand_x, self.hand_y)
        self.hand2_rect.topleft = (self.second_hand_x, self.second_hand_y)

        # Update the last_keys for the next iteration
        self.last_keys = current_keys

        for component in self.components:
            if component.picked_up_by_hand1:
                component.x = self.hand_x
                component.y = self.hand_y
            elif component.picked_up_by_hand2:
                component.x = self.second_hand_x
                component.y = self.second_hand_y

        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                self.running = False

            if event.type == pygame.KEYDOWN:
                for component in self.components:
                    # 第一只手的捡起/放下逻辑
                    if event.key == pygame.K_SPACE:
                        if not any(comp.picked_up_by_hand1 for comp in self.components):
                            if self.distance((self.hand_x, self.hand_y), component.center()) < self.pickup_threshold:
                                if not component.picked_up_by_hand1:

                                    leaf_node = self.find_leaf_by_name(self.root, component.type)
                                    if leaf_node and leaf_node.state == 'to_be_done':
                                        result = leaf_node.execute()
                                        if result == "success":
                                            message = f"{component.type} execution started successfully"
                                            self.add_message(message)
                                            component.picked_up_by_hand1 = True
                                            component.picked_up_by_hand2 = False
                                            self.states[component.type]['position'] = (component.x, component.y)
                                            self.states[component.type]['state'] = 2  # in hand1
                                            self.write_to_file()  # 更新日志文件
                                            action_code = self.get_action_code('hand1', 'execute', component.type)
                                            self.log_action('hand1', action_code)

                                        else:
                                            message = f"Failed to start execution of {component.type}"
                                            self.add_message(message)

                        elif component.picked_up_by_hand1:
                            # 放下逻辑：检查是否在机箱上方
                            if self.case_x <= self.hand_x <= self.case_x + self.case_width and \
                                    self.case_y <= self.hand_y <= self.case_y + self.case_height:
                                component.x, component.y = self.placement_positions[component.type]
                                leaf_node = self.find_leaf_by_name(self.root, component.type)
                                if leaf_node and leaf_node.state == 'doing':
                                    result = leaf_node.complete()
                                    if result == "success":
                                        message = f"{component.type} completion successful"
                                        self.add_message(message)
                                        self.states[component.type]['position'] = self.placement_positions[component.type]
                                        self.states[component.type]['state'] = 4  # in hand1
                                        self.write_to_file()  # 更新日志文件
                                        action_code = self.get_action_code('hand1', 'complete', component.type)
                                        self.log_action('hand1', action_code)
                                    else:
                                        message = f"Failed to complete {component.type}"
                                        self.add_message(message)

                                component.picked_up_by_hand1 = False
                    # 第二只手的捡起/放下逻辑
                    elif event.key == pygame.K_KP_0:
                        if not any(comp.picked_up_by_hand2 for comp in self.components):
                            if self.distance((self.second_hand_x, self.second_hand_y),
                                             component.center()) < self.pickup_threshold:
                                if not component.picked_up_by_hand2:
                                    component.picked_up_by_hand2 = True
                                    component.picked_up_by_hand1 = False
                                    message = f"{component.type} is collected"
                                    self.states[component.type]['position'] = (component.x, component.y)
                                    self.states[component.type]['state'] = 3  # in hand1
                                    self.write_to_file()  # 更新日志文件
                                    action_code = self.get_action_code('hand2', 'collect', component.type)
                                    self.log_action('hand2', action_code)
                                    self.add_message(message)
                    elif event.key == pygame.K_KP_1:
                        # 传递物品给hand1
                        if component.picked_up_by_hand2 and not any(comp.picked_up_by_hand1 for comp in self.components):
                            # 放下逻辑：检查是否在机箱上方
                            component.picked_up_by_hand1 = True
                            component.picked_up_by_hand2 = False
                            message = f"{component.type} is delivered to hand1"
                            self.states[component.type]['position'] = (component.x, component.y)
                            self.states[component.type]['state'] = 2
                            self.write_to_file()  # 更新日志文件
                            action_code = self.get_action_code('hand2', 'pass', component.type)
                            self.log_action('hand2', action_code)
                            self.add_message(message)





    def initialize_hands(self):
        self.hand_x, self.hand_y = 400, 300
        self.second_hand_x, self.second_hand_y = 200, 300

        self.hand1_image = pygame.transform.scale(pygame.image.load("./image/hand.png"), (30, 30))
        self.hand2_image = pygame.transform.scale(pygame.image.load("./image/robot.png"), (30, 30))

        self.hand1_rect = self.hand1_image.get_rect(topleft=(self.hand_x, self.hand_y))
        self.hand2_rect = self.hand2_image.get_rect(topleft=(self.second_hand_x, self.second_hand_y))
    def add_message(self, message):
        """ 添加消息及其时间戳 """
        current_time = time.time()
        self.messages.append((message, current_time))

    def update_messages(self):
        """ 更新消息列表，移除过期的消息 """
        current_time = time.time()
        self.messages = [(msg, timestamp) for msg, timestamp in self.messages if current_time - timestamp < 1]





    def run(self):
        while self.running:
            self.update()
            self.render()
            self.clock.tick(60)

        pygame.quit()

    def distance(self, point1, point2):
        return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5

    def draw_text(self, text, position, color=(255, 255, 255), font_size=64):
        font = pygame.font.SysFont(None, font_size)
        text_surface = font.render(text, True, color)
        self.screen.blit(text_surface, position)

    def render(self):
        self.screen.fill(self.BLACK)
        self.screen.blit(self.background_image, (0, 50))

        self.screen.blit(self.case_image,(self.case_x, self.case_y))

        # 绘制机箱
        # pygame.draw.rect(self.screen, self.case_color, (self.case_x, self.case_y, self.case_width, self.case_height))

        # 绘制零件
        for component in self.components:
            component.draw(self.screen)

        # 绘制手的图片
        self.screen.blit(self.hand1_image, self.hand1_rect)
        self.screen.blit(self.hand2_image, self.hand2_rect)

        executable_tasks = self.get_executable_tasks()
        if executable_tasks:
            # 显示可执行任务
            self.draw_text('Task promopt: ',(10, 10))
            for i, task in enumerate(executable_tasks):

                self.draw_text(task, (10, 10 + (i+1) * 30))
        else:
            # 如果没有可执行任务，显示完成提示
            self.draw_text("None", (10, 10))

        # 更新消息
        self.update_messages()

        # 显示消息
        for message, timestamp in self.messages:
            self.draw_text(message, (10, 500), (255, 0, 0))  # 显示在屏幕下方

        pygame.display.flip()

    def find_leaf_by_name(self, node, name):
        if node.name == name and isinstance(node, LeafNode):
            return node
        for child in node.children:
            found = self.find_leaf_by_name(child, name)
            if found:
                return found
        return None

    def write_log_header(self):
        header = "Time, " + ", ".join([f"{name}_state, {name}_position" for name in self.states.keys()])
        with open(self.log_file_path, 'w') as file:
            file.write(header + '\n')
    def create_action_log_file(self):
        with open(self.action_log_file_path, 'w') as file:
            file.write('Time, Actor, Action\n')
    def log_action(self, actor, action_code):
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        log_message = f"{current_time}, {actor}, {action_code}\n"
        with open(self.action_log_file_path, 'a') as file:
            file.write(log_message)

    def get_action_code(self, actor, action, object_type):
        object_codes = {'motherboard': 1, 'CPU': 2, 'cooler': 3, 'GPU': 4, 'memory_card': 5, 'power_supply': 6, 'hard_disk': 7, 'cover': 8}
        action_codes = {'execute': 1, 'complete': 2, 'collect': 1, 'pass': 2}
        actor_codes = {'hand1': 1, 'hand2': 2}

        return f"{actor_codes[actor]}-{action_codes[action]}{object_codes[object_type]}"




    def write_to_file(self):
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        log_message = current_time + ", " + ", ".join([f"{state['state']}, {state['position']}" for state in self.states.values()])
        with open(self.log_file_path, 'a') as file:
            file.write(log_message + '\n')




    def get_executable_tasks(self):
        # 获取可执行任务的逻辑
        incomplete_leaves = self.root.get_all_incomplete_leaf_nodes()
        following_tasks = set()

        for leaf in incomplete_leaves:
            following_tasks.update(leaf.query_following_leaf_nodes())

        executable_tasks = [leaf for leaf in incomplete_leaves if leaf not in following_tasks]
        executable_tasks = [leaf for leaf in executable_tasks if leaf.state != 'doing']

        return [leaf.name for leaf in executable_tasks]
    def initialize_tree(self):
        # 示例树


        # 创建电脑装配的树结构
        self.root = AndNode('desktop case')

        # 主板安装
        motherboard = LeafNode('motherboard')
        self.root.add_child(motherboard)

        motherboard_installation = OrNode('Partsofmainboard')
        self.root.add_child(motherboard_installation)
        cpu_fan  = AndNode('cpu|fan')
        motherboard_installation.add_child(cpu_fan)

        # CPU 安装必须在主板安装之后
        cpu_installation = LeafNode('CPU')

        fan_installation = LeafNode('cooler')
        cpu_fan.add_child(cpu_installation)
        cpu_fan.add_child(fan_installation)

        # 内存安装在主板安装之后
        ram_installation = LeafNode('memory_card')
        motherboard_installation.add_child(ram_installation)





        # 显卡安装在主板安装之后
        gpu_installation = LeafNode('GPU')
        motherboard_installation.add_child(gpu_installation)
        others_installation = OrNode('others')
        self.root.add_child(others_installation)

        # 硬盘安装在主板安装之后
        storage_installation = LeafNode('hard_disk')
        others_installation.add_child(storage_installation)


        # 电源安装应该在所有主板组件都已安装后进行
        # 但为了表达这一点，我们将它放置在最后一个位置
        psu_installation = LeafNode('power_supply')
        others_installation.add_child(psu_installation)

        lid_installation =LeafNode('cover')
        self.root.add_child((lid_installation))



# 定义零件类
class Component:
    def __init__(self, x, y, width, height, type,image_path):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.type = type
        self.original_image = pygame.image.load(image_path)
        self.image = pygame.transform.scale(self.original_image, (width, height))
        self.rect = self.image.get_rect(topleft=(x, y))
        self.picked_up_by_hand1 = False
        self.picked_up_by_hand2 = False

    def center(self):
        return center_position(self.x, self.y, self.width, self.height)

    def draw(self, screen):
        self.rect = self.image.get_rect(topleft=(self.x, self.y))
        screen.blit(self.image, self.rect)


class Node:
    def __init__(self, name):
        self.name = name
        self.children = []
        self.parent = None

    def add_child(self, child):
        child.parent = self
        self.children.append(child)


    # 新增一个方法来找到任意深度的祖先节点
    def get_ancestor(self, i):
        if i == 0:
            return self
        if self.parent:
            return self.parent.get_ancestor(i - 1)
        return None
    # 获取当前节点的所有后续兄弟节点
    def get_following_siblings(self):
        if not self.parent:
            return []
        siblings = self.parent.children
        index = siblings.index(self)
        return siblings[index+1:]
    # 获取当前节点的所有叶节点子孙
    def get_all_leaf_nodes(self):
        if isinstance(self, LeafNode):
            return [self]
        leaf_nodes = []
        for child in self.children:
            leaf_nodes.extend(child.get_all_leaf_nodes())
        return leaf_nodes

    def get_all_incomplete_leaf_nodes(self):
        incomplete_leaf_nodes = []
        if isinstance(self, LeafNode) and self.state != 'done':
            incomplete_leaf_nodes.append(self)
        else:
            for child in self.children:
                incomplete_leaf_nodes.extend(child.get_all_incomplete_leaf_nodes())
        return incomplete_leaf_nodes
    def get_all_executing_leaf_nodes(self):
        incomplete_leaf_nodes = []
        if isinstance(self, LeafNode) and self.state == 'doing':
            incomplete_leaf_nodes.append(self)
        else:
            for child in self.children:
                incomplete_leaf_nodes.extend(child.get_all_executing_leaf_nodes())
        return incomplete_leaf_nodes

class AndNode(Node):
    def __init__(self, name):
        super().__init__(name)

class OrNode(Node):
    def __init__(self, name):
        super().__init__(name)

class LeafNode(Node):
    def __init__(self, name):
        super().__init__(name)
        self.state = 'to_be_done'

    def execute(self):
        try:
            if self.can_execute():
                self.state = 'doing'
                print(f"{self.name} doing")
                return "success"
            else:
                Exception(f"无法执行{self.name}，因为并未满足先决条件。")
        except Exception as e:
            print(e)
            return "failure"



    def complete(self):
        try:
            if self.state != 'doing':
                raise Exception(f"无法完成{self.name}，因为它尚未开始执行。")
            self.state = 'done'
            print(f"{self.name} 已完成")
            return "success"
        except Exception as e:
            print(e)
            return "failure"

    def can_execute(self):
        # 检查所有祖先节点，以确保满足先决条件
        ancestor = self.parent
        ancestor_child= self
        while ancestor:
            # if ancestor.name == 'root':
            #     return True

            if isinstance(ancestor, AndNode):
                prev_sibling = self.get_previous_sibling_at_level(ancestor,ancestor_child)
                if prev_sibling and not all_leaf_nodes_completed(prev_sibling):
                    return False
            ancestor_child = ancestor
            ancestor = ancestor.parent
        return True
    # 得到当前节点的层级
    def get_level(self):
        level = 0
        current = self
        while current.parent:
            level += 1
            current = current.parent
        return level
    # 新增查询特定深度的祖先节点的后续兄弟节点中的叶节点的功能
    def query_following_leaf_nodes(self):
        following_leaf_nodes = []
        ancestor = self.parent
        ancestor_child  = self
        while ancestor:
            # 如果祖先是AndNode，获取下一代的所有后续兄弟节点
            if isinstance(ancestor, AndNode):
                following_siblings = ancestor_child.get_following_siblings()
                for sibling in following_siblings:
                    # 添加sibling的叶节点子孙到结果列表中
                    following_leaf_nodes.extend(sibling.get_all_leaf_nodes())
            ancestor_child = ancestor
            ancestor = ancestor.parent
        return following_leaf_nodes

    # 找到在指定层级的前一个兄弟节点
    def get_previous_sibling_at_level(self, ancestor,ancestor_child):
        # parent = self.get_ancestor(self.get_level() - ancestor.get_level())
        parent = ancestor
        index = parent.children.index(ancestor_child)
        if index > 0:
            return parent.children[index - 1]
        return None


def all_leaf_nodes_completed(node):
    if isinstance(node, LeafNode):
        return node.state == 'done'
    return all(all_leaf_nodes_completed(child) for child in node.children)
def check_completion(node):
    if isinstance(node, LeafNode) and node.state != 'done':
        return False
    for child in node.children:
        if not check_completion(child):
            return False
    return True

# 定义手和主板的中心点位置
def center_position(x, y, width, height):
    return x + width // 2, y + height // 2




if __name__ == "__main__":
    game = ComputerAssemblyGame()
    game.run()