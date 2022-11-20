import numpy as np 

class associative_memory : 
        
    def __init__(self, delta_plus=1, delta_neg=0.1):
        self.delta_plus       = delta_plus
        self.delta_neg        = delta_neg
        self.arry             = []
        self.node_dict        = dict()
        self.label_dict       = dict()

        
    def is_arry_empty(self):
        if not self.arry:
            return True
        return False

    def is_label_exists(self, label_id) : 
        if(label_id in self.label_dict) : 
            return True
        else :
            return False

    def is_node_exists(self, node_id) : 
        #print("node dict : ", self.node_dict, node_id)
        if(node_id in self.node_dict) : 
            return True
        return False
        
    def get_label_idx(self, label_id) : 
        return self.label_dict[label_id]
    
    def get_node_idx(self, node_id) : 
#         print(node_id)
        return self.node_dict[node_id]
            
    def get_current_label_count(self):
        if(self.is_arry_empty()) :
            return 0
        else :
            return len(self.arry[0])
    
    def get_current_node_count(self) : 
        if(self.is_arry_empty()) :
            return 0
        else :
            return len(self.arry)
    
    def add_node(self, node_id):
        # print("******** Add Node **************", node_id)
        # raw_input()
        if(self.is_arry_empty()): 
            row = [] 
            self.node_dict[node_id] = 0
        else : 
            row = [0] * len(self.arry[0])
            self.node_dict[node_id] = len(self.arry)
        self.arry.append(row)

    def add_label(self, node_id, label_id):
        # print("******** Add Label **************")
        label_idx = self.get_current_label_count()
        self.label_dict[label_id] = label_idx
#         print(label_idx)
        node_idx = self.get_node_idx(node_id)
        for i in range(len(self.arry)) : 
            self.arry[i].append(0)
        
        # if(max(self.arry[node_idx]) > 5) :
        #     # print(self.delta_plus, self.delta_neg)
        #     node_idx_max_label = np.argmax(self.arry[node_idx])
        #     print("Node Idx Max Label : ", node_idx_max_label)
        #     label_array = np.array(self.arry)
        #     label_array = label_array[:,node_idx_max_label]
        #     label_max_node_idx = np.argmax(label_array)
        #     # print(label_array)
        #     # print("!!!!!!!!!!!!!!!!!!!!!!!!Current max label before you erase it : node : %d value : %f, label : %d max node for label : %d" %(node_idx, max(self.arry[node_idx]), self.arry[node_idx].index(max(self.arry[node_idx])), label_max_node_idx))
        # else  :
        #     self.arry[node_idx]           = [0] * self.get_current_label_count()
        #     self.arry[node_idx][label_idx] = 1

        for i in range(len(self.arry[node_idx])):
            if(i==label_idx): 
                self.arry[node_idx][i] += self.delta_plus
            else : 
                self.arry[node_idx][i] -= self.delta_neg
                if(self.arry[node_idx][i] < 0) :
                    self.arry[node_idx][i] = 0
        
    def update_label(self, node_id, label_id):
        # print("******** Update Label **************")
        label_idx = self.get_label_idx(label_id)
        node_idx = self.get_node_idx(node_id)
        for i in range(len(self.arry[node_idx])):
            if(i==label_idx): 
                self.arry[node_idx][i] += self.delta_plus
            else : 
                self.arry[node_idx][i] -= self.delta_neg
                if(self.arry[node_idx][i] < 0) :
                    self.arry[node_idx][i] = 0
                    
    def train_mem(self, node_id, label_id):
#         if(node_id > self.get_current_node_count()-1) :
#             self.add_node()
        if not(self.is_node_exists(node_id)):
            self.add_node(node_id)
#         self.print_mem
        if(self.is_label_exists(label_id)) :
            self.update_label(node_id, label_id)
        else :
            self.add_label(node_id, label_id)
            
    def print_mem(self) :
        print("*********************Associative Matrix*************************")
        for i in range(len(self.arry)) : 
            print(self.arry[i])
        print("*******************Label-Column Dictionary**********************")
        print(self.label_dict)
        # print("**********************Node Dictionary**********************")
        # print(self.node_dict)
        
    def get_label(self, node_id):
        # print("get label")
#         if(node_id >= len(self.arry)) :
#         print(self.node_dict)
        if not(self.is_node_exists(node_id)):
            # print("Assoc Matrix : Node %d, out of range" %(node_id))
            return -1
        else :
            node_idx = self.get_node_idx(node_id)
            label_idx = self.arry[node_idx].index(max(self.arry[node_idx]))
            label_id = list(self.label_dict.keys())[list(self.label_dict.values()).index(label_idx)]
#             print(label_idx)
            return label_id
