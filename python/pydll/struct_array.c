#include <stdio.h>

typedef void (*Fun)(int);
Fun p = NULL;

void register_callback(Fun pCallback)
{
    p = pCallback;
}

int cpp_space_init()
{
    (*p)(32);
    return 1;
}


typedef struct student{
    char name;
    short class;
    double num;
    int age;
}stu;

typedef struct stu_struct_array{
    stu stu_array[2];
}stu_struct_array_t;

int stu_test(stu *msg){
    printf("stu name: %d\n", msg->name);
    printf("stu class: %d\n", msg->class);
    printf("stu num: %f\n", msg->num);
    printf("stu age: %d\n", msg->age);
    return 1;
}


int struct_test(stu *msg, stu_struct_array_t *nest_msg, char *buff){

    int index = 0;

    printf("stu name: %d\n", msg->name);
    printf("stu class: %d\n", msg->class);
    printf("stu num: %f\n", msg->num);
    printf("stu age: %d\n", msg->age);

    memcpy(nest_msg->stu_array, msg, sizeof(stu));

    printf("stu array index 0 name: %d\n", nest_msg->stu_array[0].name);
    printf("stu array index 0 class: %d\n", nest_msg->stu_array[0].class);

    memcpy(buff, msg, sizeof(stu));
    printf("buff: %d %d", buff[0], buff[1]);

    return 1;
}