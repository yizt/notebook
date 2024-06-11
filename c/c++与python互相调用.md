[TOC]

### 数值类型

```python
double_values = (c_double * 17)(*[self.value_dict["ir_scale"],
                                          self.value_dict["second_ir_scale"],
                                          self.value_dict["angle_scale"],
                                          self.value_dict["ir_kps_distance"],
                                          self.value_dict["second_ir_kps_distance"],
                                          self.value_dict["m00"],
                                          self.value_dict["m01"],
                                          self.value_dict["m02"],
                                          self.value_dict["m10"],
                                          self.value_dict["m11"],
                                          self.value_dict["m12"],
                                          self.value_dict["e00"],
                                          self.value_dict["e01"],
                                          self.value_dict["e02"],
                                          self.value_dict["e10"],
                                          self.value_dict["e11"],
                                          self.value_dict["e12"]
                                          ])
self.so.set_double_values(pointer(double_values))
```



```c++
void set_double_values(double *p_double_vals);   // 设置double类型值

void set_double_values(double *p_double_vals)   // 设置double类型值
{
    ir_scale = *(p_double_vals+0);
    second_ir_scale = *(p_double_vals+1);
    angle_scale = *(p_double_vals+2);

    ir_kps_distance = *(p_double_vals+3);
    second_ir_kps_distance = *(p_double_vals+4);

    int i = 5;
    m00 = *(p_double_vals+i);
    m01 = *(p_double_vals+i+1);
    m02 = *(p_double_vals+i+2);
    m10 = *(p_double_vals+i+3);
    m11 = *(p_double_vals+i+4);
    m12 = *(p_double_vals+i+5);

    i = 11;
    e00 = *(p_double_vals+i);
    e01 = *(p_double_vals+i+1);
    e02 = *(p_double_vals+i+2);
    e10 = *(p_double_vals+i+3);
    e11 = *(p_double_vals+i+4);
    e12 = *(p_double_vals+i+5);
}
```





### 字符类型

第一种char*

```c++
msg = "{}:{}".format(step_name, self.exception_type_dict[exception])
self.so.change_warning(c_int(1), c_char_p(msg.encode("utf-8")))
```



```c++
static QString warning_info;

void change_warning(int val1, char*val2)
{
    warning_state = val1;
    warning_info = val2;
    warning_flag = 1;
}
```



第二种

```python
self.so.set_rtmp_addr(c_char_p(self.rtmp_url.encode("utf-8")),
                              c_int(len(self.rtmp_url)))
```



```c++
char rtmp_addr[512]; //视频推流地址

void set_rtmp_addr(char* p_rtmp_addr, int len){
    cout << "set_rtmp_addr p_rtmp_addr :" << p_rtmp_addr << "; len:" << len << endl;
    memcpy(rtmp_addr, p_rtmp_addr, len);
}
```



### 数据结构

1. python调用C++

```python
class AreaProps(Structure):
    """接头区域属性信息"""
    _fields_ = [("center_x", c_int),
                ("center_y", c_int),
                ("radius", c_int),
                ("center_ratio", c_double),
                ("angle_ratio", c_double)]

    def to_str(self):
        return {"center_x": self.center_x,
                "center_y": self.center_y,
                "radius": self.radius,
                "center_ratio": self.center_ratio,
                "angle_ratio": self.angle_ratio
                }
    
def to_area_properties(self):
    """
        接头朝向测量的属性配置
        :return:
        """
    props = self.area_props
    area_list = []
    area_props_list_struct = AreaProps * len(props)
    for prop in props:
        val_list = [c_int(prop[0]),
                    c_int(prop[1]),
                    c_int(prop[2]),
                    c_double(prop[3]),
                    c_double(prop[4])]
        area_list.append(AreaProps(*val_list))
        return AreaPropsList(area_props_list_struct(*area_list))

self.so.set_area_props(pointer(self.to_area_properties()))
```



```c++
/**
 * 接头属性，朝向测量使用 
 */
typedef struct area_props
{
    int center_x;
    int center_y;
    int radius;
    double center_ratio;
    double angle_ratio;
} area_props;
/**
 * 接头属性数组，朝向测量使用 
 */
typedef struct area_props_array
{
    area_props area_props_list[9];
} area_props_array;

void set_area_props(area_props_array* p_area_props)  // 接头区域的属性
{
    for(int i=0;i<PATTERN_AREA_NUM;i++){
        area_properies[i][0]=p_area_props->area_props_list[i].center_x;
        area_properies[i][1]=p_area_props->area_props_list[i].center_y;
        area_properies[i][2]=p_area_props->area_props_list[i].radius;
        area_properies[i][3]=p_area_props->area_props_list[i].center_ratio;
        area_properies[i][4]=p_area_props->area_props_list[i].angle_ratio;
    }
}
```



2. c++ 调用python

```c++
typedef struct area
{
    int x1;
    int y1;
    int x2;
    int y2;
} area;

typedef struct area_array
{
    area area_list[9];
} area_array;

typedef void (*FunRgt)(area_array);
FunRgt pRgt = NULL;

// 函数调用
area_array* p_area_list

for (int j = 0; j < PATTERN_AREA_NUM; j++)
    {
        p_area_list->area_list[j].x1 = dst_area[j][0];
        p_area_list->area_list[j].y1 = dst_area[j][1];
        p_area_list->area_list[j].x2 = dst_area[j][2];
        p_area_list->area_list[j].y2 = dst_area[j][3];
	printf("[%d, %d, %d, %d]\n", dst_area[j][0], dst_area[j][1], dst_area[j][2], dst_area[j][3]);
    }

(*pRgt)(*p_area_list);
```



```python
class Area(Structure):
    """
    坐标区域
    """
    # _fields_是容纳每个结构体成员类型和值的列表，可以配合自动生成fields list和value list的函数使用
    _fields_ = [("x1", c_int),
                ("y1", c_int),
                ("x2", c_int),
                ("y2", c_int)]

    def to_str(self):
        return [self.x1, self.y1, self.x2, self.y2]

class AreaList(Structure):
    """坐标区域列表"""
    _fields_ = [("area_list", Area * 9)]

    def to_str(self):
        return [area.to_str() for _, area in enumerate(self.area_list)]


def rgt_callback(self, areas: AreaList):
    """
        位置校准c回调函数
        :param areas:
        :return:
        """
    self.state.on_image_registration(areas)
    self.state.on_step_guide()
    def on_image_registration(self, rgt_areas: AreaList):
        """
        位置配准回调:
        :param rgt_areas: 配准后的坐标区域更新
        :return:
        """
        area_list = [[area.x1, area.y1, area.x2, area.y2] for i, area in
                     enumerate(rgt_areas.area_list)]
        for step in self.step_list:
            area_idx = step["stepAreaIdx"]
            step["stepArea"] = area_list[area_idx]
        self.rgt_finish_flag = True
        self.debug("on_image_registration: self.step_list:\n{}".format(self.step_list))
        self.debug("on_image_registration: rgt_areas:\n{}".format(rgt_areas.to_str()))
        
## 调用时传入回调函数        
p_area_list = pointer(area_list)
c_func = CFUNCTYPE(None, AreaList)
self.so.start_image_registration(p_area_list, c_func(self.rgt_callback))
```



