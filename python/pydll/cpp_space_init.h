#ifndef CPP_SPACE_INIT_H
#define CPP_SPACE_INIT_H



#ifdef __cplusplus
extern "C" {
#endif

typedef void (*Fun)(int);
Fun p = NULL;

void regist_callback(Fun pCallback);
int cpp_space_init();


#ifdef __cplusplus
};
#endif
#endif // CPP_SPACE_INIT_H