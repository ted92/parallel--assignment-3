__global__ string guess_password(int max_length, string* data_in, string known)
{
	guesses = collections.deque(string.printable)

    while(guesses){
        cur_guess = guesses.popleft();
        if(len(cur_guess) > max_length){
            return FALSE;
        }
        if(try_password(in_data, cur_guess, known_part)){
            return cur_guess;
        }
        else{
            if(len(cur_guess) != max_length){
                int i;
                char c[] = string.printable;
                for (i = 0; i < strlen(string.printable); i++){
                    guesses.append(cur_guess + c[i]);
                }
            }
        }
    }
}