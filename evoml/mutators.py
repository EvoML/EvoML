

def segment_mutator(individual, pool_data, indpb, delpb = 0.33, addpb = 0.66):
    """
    Takes data from pool_data and mutuates each dataframe
    in the individual.

    Parameters:
    ------------
    individual : list of DataFrames

    pool_data : DataFrame
        Pool data from which rows are added or swapped.

    indpb : float, required
        Probablity with which mutuation happens on each DataFrame of the
        individual.

    delpb : float, optional, default, 0.33
        Probablity with which rows are deleted in a mutuation

    addpb : float, optional, default, 0.33
        Probablity with which rows are added from the pool_data in a mutuation

    

    Mutate can be:
     - add rows from pool_data randomly
     - delete rows randomly from the individual
     - replace a few rows from that of df 

    """


    df_train = pool_data
    
    for i, df_ in enumerate(individual):
        if random.random()>=indpb:
            continue
        # play around with tenpercent of current data.
        
        #todo: Parameterize this.
        n_rows = int(0.2*df_.shape[0])
        rnd = random.random()
        if rnd<0.33:
            #add rows from the main df
            
            rows = np.random.choice(df_train.index.values, n_rows)
            df_ = df_.append(df_train.ix[rows])
        elif rnd<0.66:
            # delete rows randomly from the individual
            df_.drop(labels=np.random.choice(df_.index, n_rows), axis=0, inplace=True)
        else:
            #replace a few rows
            df_.drop(labels=np.random.choice(df_.index, 1), axis=0, inplace=True)
            rows = np.random.choice(df_train.index.values, n_rows)
            df_ = df_.append(df_train.ix[rows])
        
        individual[i] = df_.copy()

    
    return (individual,)
