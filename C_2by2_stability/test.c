/*
  Stability: Statistical version in C

  Future plans: Try to trace in each brunch the difference of
                teacher and learner, then do some statistics
                such as expectation / standard deviation / histo.
                This can be done in dfs, but we need a place to
                store them, not hard.

  Difference: Now I am using the last position of each file to store the
              head-to-head difference of distribution, i.e., the
              1-norms in each step. Not Wasserstein.

  New tests: use this to generate something? Verify Linearity of log(u_h/(1-u(h)))
             stability of (1-u(h))/u(h)?
*/

#include <time.h>
#include <pthread.h>
#include "test.h"
// >>> configurations >>>
// distribution histo resolution.
#define RESOLUTION 10000
// first round depth, 64 threads in the second round.
#define DEPTH_INIT 6
// total depth.
#define DEPTH_FINAL 39
#define PREFIX "./result5/Test_FIVE_"
// factor of perturbation ratio
#define PERTURB 0
// 2E-3
#define ABS(a) ((a)>0 ? (a) : -(a))
// <<< configurations <<<

double joint_teach[2][2]={{0.5, 0.8},
                          {0.5, 0.2}};
double joint_learn[2][2]={{0, 0},
                          {0, 0}};
double prior_teach[2]={0.3, 0.7};
double prior_learn[2]={0, 0};

typedef struct status_struct{
    double teach;
    double learn;
    double prob;
}Status;


typedef struct node_struct{
    Status status;
    int depth_init;
    int depth_final;
    char prefix[50];
    FILE * backup;
}Node;


long dfs(Status status, double** record_t, double** record_l,
         long count, FILE* backup)
/*
  DFS recursions

  Parameters:
  -----------
  status: current status
  record_t: record array (teach) of statistical array
  record_l: record array (learn) of statistical array
            Note: we always write in the top of records, which
                  means in the dfs we pass to the next layer
                  the next pointer in record_?[], if it ends,
                  the dfs reaches max depth and stops going deeper.
  count   : a long integer for counting how many nodes are
            processed
  backup  : only used in first round, record raw leaf-data for
            constructing roots in threading round.
            It is NULL in the threading round and not used.
*/
{
    long ret=count + 1;
    Status status_L, status_R;
    double T_res[2], L_res[2];

    // calculate teaching results, two different possibilities
    teaching(joint_teach, joint_learn,
             status.teach, status.learn,
             T_res, L_res);
    status_L.teach = T_res[0];
    status_L.learn = L_res[0];
    status_L.prob  = status.prob * T_res[0] / (T_res[0]+T_res[1]);
    status_R.teach = T_res[1];
    status_R.learn = L_res[1];
    status_R.prob  = status.prob * T_res[1] / (T_res[0]+T_res[1]);

    // write data into record_?, just in the top row, this uses
    // the columns: 0 ~ RESOLUTION.
    (*record_t)[(int)(RESOLUTION*status_L.teach)] += status_L.prob;
    (*record_l)[(int)(RESOLUTION*status_L.learn)] += status_L.prob;

    // printf("%d, %lf, %lf\n", (int)(RESOLUTION*status_L.learn), status_L.learn, status_L.prob);

    (*record_t)[(int)(RESOLUTION*status_R.teach)] += status_R.prob;
    (*record_l)[(int)(RESOLUTION*status_R.learn)] += status_R.prob;

    // make use of record_?[RESOLUTION+1]
    (*record_t)[RESOLUTION + 1] += ABS(status_L.teach - status_L.learn) * status_L.prob;
    (*record_t)[RESOLUTION + 1] += ABS(status_R.teach - status_R.learn) * status_R.prob;
    (*record_l)[RESOLUTION + 1] += ABS(status_L.teach - status_L.learn) * status_L.prob;
    (*record_l)[RESOLUTION + 1] += ABS(status_R.teach - status_R.learn) * status_R.prob;

    if(*(record_t+1)){ // check whether reaches the max depth
        ret = dfs(status_L, record_t+1, record_l+1, ret, backup);
        ret = dfs(status_R, record_t+1, record_l+1, ret, backup);
    }
    else if(backup){ // if reaches max depth and backup is not NULL
        fwrite(&status_L,sizeof(status_L),1,backup);
        fwrite(&status_R,sizeof(status_R),1,backup);
    }
    return ret;
}


long calculate(char prefix[], Status status, int depth_init,
               int depth_final, FILE * backup)
/*
  Parameters:
  -----------
  prefix     : a string of filename prefix.
  status     : Status struct denote the priors and initial probability
               of the root node
  depth_init : the depth of root node (0 in the first round, DEPTH_INIT
               in the threading round)
  depth_final: depth of leaf nodes (DEPTH_INIT in first round, DEPTH_FINAL
               in threading)
  backup     : file stream of raw leaf-data, used to construct root nodes
               in threading round, NULL in threading round
*/
{
    long count=0;

    double ** record_t;
    double ** record_l;
    // allocate record_t record_l
    record_t = (double**) malloc(sizeof(double*)*(depth_final - depth_init + 1));
    record_l = (double**) malloc(sizeof(double*)*(depth_final - depth_init + 1));

    record_t[depth_final - depth_init] = NULL;
    record_l[depth_final - depth_init] = NULL;


    // allocate and initialzie data arrays
    for(int i=depth_init; i<depth_final; i++){
        record_t[i - depth_init] = (double*) malloc((RESOLUTION+2)*sizeof(double));
        record_l[i - depth_init] = (double*) malloc((RESOLUTION+2)*sizeof(double));
        memset(record_t[i - depth_init],0,(RESOLUTION+2)*sizeof(double));
        memset(record_l[i - depth_init],0,(RESOLUTION+2)*sizeof(double));
    }

    // calculate and save in data arrays
    count = dfs(status, record_t, record_l, count, backup);

    char* filename;
    int filename_len;
    filename_len = strlen(prefix)+30;
    filename = (char*)malloc(filename_len);
    FILE * fp;

    // record and free the data arrays
    for(int i=depth_init; i<depth_final; i++){

        sprintf(filename, "%slvl_%02d_T.log", prefix, i);
        fp = fopen(filename,"wb");       // open fp
        fwrite(record_t[i - depth_init], // initial pointer
               sizeof(double),           // size
               RESOLUTION + 2,           // count
               fp);                      // stream
        fclose(fp);                      // close fp
        free(record_t[i - depth_init]);

        sprintf(filename, "%slvl_%02d_L.log", prefix, i);
        fp = fopen(filename,"wb");       // open fp
        fwrite(record_l[i - depth_init], // initial pointer
               sizeof(double),           // size
               RESOLUTION + 2,           // count
               fp);                      // stream
        fclose(fp);                      // close fp
        free(record_l[i - depth_init]);
    }
    // free record_t record_l
    free(record_t);
    free(record_l);
    free(filename);
    return count;
}


void* thread_shell(void* node)
{
    Node* root;
    root = (Node*)node;
    long ret;
    ret = calculate(root->prefix, root->status, root->depth_init,
                    root->depth_final, root->backup);
    // printf("%d\n",ret);
    return (void*) ret;
}


double std_diff_random(int index){
    return (double) (rand() - RAND_MAX / 2) / RAND_MAX * PERTURB;
}

double zero_random(int index) {return (double) 0;}

double pos_constant(int index) {return (double) 0.4;}

double neg_constant(int index) {return (double) -0.4;}

void init(double rand_mat(int index), double rand_prior(int index))
/* Initialization */
{
    srand(time(0));

    char filename_conf[40];
    sprintf(filename_conf, "%sconfig.log", PREFIX);
    FILE *fp;
    fp = fopen(filename_conf,"wb");
    for(int i=0; i<2; i++){
        for(int j=0; j<2; j++){
            joint_learn[i][j] = joint_teach[i][j] + rand_mat(i*2+j);
        }
    }
    prior_learn[0] = prior_teach[0] + rand_prior(4);
    prior_learn[1] = 1.0 - prior_learn[0];
    printf("Learner's Prior: [%lf, %lf]\n",prior_learn[0],prior_learn[1]);

    fwrite(&joint_teach,sizeof(double),2,fp);
    fwrite(joint_teach+1,sizeof(double),2,fp);
    fwrite(joint_learn[0],sizeof(double),2,fp);
    fwrite(joint_learn[1],sizeof(double),2,fp);
    fwrite(prior_teach,sizeof(double),2,fp);
    fwrite(prior_learn,sizeof(double),2,fp);

    fclose(fp);
    printf("initialized\n"); // for debug
}

int main (int argc, char** argv)
{
    clock_t time = clock();
    FILE * backup;
    long count=0;
    char backup_filename[50];
    sprintf(backup_filename, "%s%s.log", PREFIX, "thread_root_backup");
    backup = fopen(backup_filename, "wb");
    // initialization
    init(zero_random, pos_constant);

    // first round
    Node node;
    node.depth_init = 0;
    node.depth_final = DEPTH_INIT;
    node.backup = backup;
    node.status.teach = prior_teach[0];
    node.status.learn = prior_learn[0];
    node.status.prob  = 1.0;
    strcpy(node.prefix,PREFIX);

    count = (long)thread_shell(&node);

    fclose(backup);
    printf("first round finished\n"); // for debug
    // data for threading

    Status leaf_status[1<<DEPTH_INIT];
    backup = fopen(backup_filename, "rb");
    fread(leaf_status, sizeof(Status), 1<<DEPTH_INIT, backup);
    fclose(backup);

    // second (threading) round
    pthread_t threads[1<<DEPTH_INIT];

    node.depth_init = DEPTH_INIT;
    node.depth_final = DEPTH_FINAL;
    node.backup = NULL;

    Node thread_nodes[1<<DEPTH_INIT];
    for(int i=0; i < (1<<DEPTH_INIT); i++){
        int create_check;
        memcpy(&(thread_nodes[i]), &node, sizeof(Node));
        thread_nodes[i].status = leaf_status[i];
        sprintf(thread_nodes[i].prefix, "%sth_%d", PREFIX, i);
        // printf("%s\n",thread_nodes[i].prefix);
        create_check = pthread_create(threads+i, NULL, thread_shell, &(thread_nodes[i]));
        // printf("%d:%d\n",i,create_check);
    }
    printf("all threads created\n"); // for debug


    long thread_return[1<<DEPTH_INIT]={0,};
    for(int i=0; i < (1<<DEPTH_INIT); i++){
        printf("waiting for thread %d...\n",i);
        // pthread_join(threads[i], NULL);
        pthread_join(threads[i], (void*) &thread_return[i]);
        count += thread_return[i];
    }


    printf("Finished simulation with dfs nodes called %ld times.\n", count);
    // measuring time used.
    time = clock() - time;
    printf("Time used: %ld : %ld : %f.\n",
           (long)time/CLOCKS_PER_SEC/3600,
           ((long)time/CLOCKS_PER_SEC/60)%60,
           (int)(time/CLOCKS_PER_SEC)%60+(float)(time%CLOCKS_PER_SEC)/CLOCKS_PER_SEC);
    return 0;
}
