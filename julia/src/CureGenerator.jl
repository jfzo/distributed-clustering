function generate(;big_circle::Int64=2000, left_ellipse::Int64=2000, right_ellipse::Int64=2000, upper_circle::Int64=2000, lower_circle::Int64=2000, noisy_bridge::Int64=2000, back_noise::Int64=2000)
    #=
    n1 = 2000; # Big circle
    n2 = 1000; # 1st elipse
    n6 = 500;   # Noisy bridge
    n3 = 1500; # 2nd elipse
    n4 = 500; # upper small circle
    n5 = 600; # lower small circle
    n7 = 3000;  # background noise
    =#
    n1 = big_circle;
    n2 = left_ellipse;
    n3 = right_ellipse;
    n4 = upper_circle;
    n5 = lower_circle;
    n6 = noisy_bridge;    
    n7 = back_noise;
    
    
    # The big circle
    mu = [0 ;-0.5];
    Sigma = [5 0; 0 1]; R = chol(Sigma);
    z1 = repmat(mu,1,n1) + R*randn(2,n1);
    l1 = zeros(1,n1) + 1;

    # the 1st elipse
    mu = [-2 ;6];
    Sigma = [3 0; 0 0.05]; R = chol(Sigma);
    z2 = repmat(mu,1,n2) + R*randn(2,n2);
    l2 = zeros(1,n2) + 2;


    # Noisy bridge
    mu = [3.0; 5.9];
    Sigma = [20 0; 0 0.01]; R = chol(Sigma);
    z6 = repmat(mu,1, n6) + R*rand(2, n6);
    l6 = zeros(1, n6) + 6;

    # the 2nd elipse
    mu = [12; 6];
    Sigma = [3 0; 0 0.05]; R = chol(Sigma);
    z3 = repmat(mu,1,n3) + R*randn(2, n3);
    l3 = zeros(1, n3) + 3;

    # the upper small circle
    mu = [15; 1];
    Sigma = [0.3 0; 0 0.05]; R = chol(Sigma);
    z4 = repmat(mu,1, n4) + R*randn(2, n4);
    l4 = zeros(1, n4) + 4;

    # the lower small circle
    mu = [15; -2];
    Sigma = [0.3 0; 0 0.05]; R = chol(Sigma);
    z5 = repmat(mu,1, n5) + R*randn(2, n5);
    l5 = zeros(1, n5) + 5;

    # background noise
    mu = [-8 ;-5];
    Sigma = [700 0; 0 150]; R = chol(Sigma);
    z7 = repmat(mu,1, n7) + R*rand(2, n7);
    l7 = zeros(1, n7) + 7;

    # merging 
    Z = hcat(z1, z2, z3, z4, z5,z6,z7);
    labels = hcat(l1,l2,l3,l4,l5,l6,l7);

    newZ = vcat(Z,labels);
end