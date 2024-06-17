Require Import ZArith.
From QuickChick Require Import QuickChick.
From ExtLib Require Import Monad.
From ExtLib.Data.Monads Require Import OptionMonad.
Import QcNotation.
Import MonadNotation.
From Coq Require Import List.
Import ListNotations.

From RBT Require Import Impl Spec.

Inductive CtorColorKVTree :=
  | CtorColorKVTree_E
  | CtorColorKVTree_T.

Inductive LeafCtorColorKVTree :=
  | LeafCtorColorKVTree_E.

Inductive CtorColor :=
  | CtorColor_R
  | CtorColor_B.

Inductive LeafCtorColor :=
  | LeafCtorColor_R
  | LeafCtorColor_B.

Inductive TupLeafCtorColorCtorColorKVTreeCtorColorKVTree :=
  | MkLeafCtorColorCtorColorKVTreeCtorColorKVTree : LeafCtorColor -> CtorColorKVTree -> CtorColorKVTree -> TupLeafCtorColorCtorColorKVTreeCtorColorKVTree.

Inductive TupLeafCtorColorLeafCtorColorKVTreeLeafCtorColorKVTree :=
  | MkLeafCtorColorLeafCtorColorKVTreeLeafCtorColorKVTree : LeafCtorColor -> LeafCtorColorKVTree -> LeafCtorColorKVTree -> TupLeafCtorColorLeafCtorColorKVTreeLeafCtorColorKVTree.

Definition genLeafColor (chosen_ctor : LeafCtorColor) (stack1 : nat) (stack2 : nat) : G (Color) :=
  match chosen_ctor with
  | LeafCtorColor_R  => 
    (returnGen (R ))
  | LeafCtorColor_B  => 
    (returnGen (B ))
  end.

Definition genLeafTree (chosen_ctor : LeafCtorColorKVTree) (stack1 : nat) (stack2 : nat) : G (Tree) :=
  match chosen_ctor with
  | LeafCtorColorKVTree_E  => 
    (returnGen (E ))
  end.

Fixpoint genTree (size : nat) (chosen_ctor : CtorColorKVTree) (stack1 : nat) (stack2 : nat) : G (Tree) :=
  match size with
  | O  => match chosen_ctor with
    | CtorColorKVTree_E  => 
      (returnGen (E ))
    | CtorColorKVTree_T  => 
      (bindGen 
      (* Frequency2 *) (freq [
        (* 1 *) (match (stack1, stack2) with
        | (4, 4) => 50
        | (4, 6) => 50
        | (6, 4) => 50
        | (6, 6) => 50
        | _ => 500
        end,
        (returnGen (MkLeafCtorColorLeafCtorColorKVTreeLeafCtorColorKVTree (LeafCtorColor_R ) (LeafCtorColorKVTree_E ) (LeafCtorColorKVTree_E )))); 
        (* 2 *) (match (stack1, stack2) with
        | (4, 4) => 50
        | (4, 6) => 50
        | (6, 4) => 50
        | (6, 6) => 50
        | _ => 500
        end,
        (returnGen (MkLeafCtorColorLeafCtorColorKVTreeLeafCtorColorKVTree (LeafCtorColor_B ) (LeafCtorColorKVTree_E ) (LeafCtorColorKVTree_E ))))]) 
      (fun param_variantis => (let '(MkLeafCtorColorLeafCtorColorKVTreeLeafCtorColorKVTree ctor1 ctor2 ctor3) := param_variantis in

        (bindGen (genLeafColor ctor1 stack2 1) 
        (fun p1 => 
          (bindGen (genLeafTree ctor2 stack2 3) 
          (fun p2 => 
            (bindGen 
            (* GenZ1 *)
            (let _weight_1 := match (stack1, stack2) with
            | (4, 4) => 50
            | (4, 6) => 50
            | (6, 4) => 50
            | (6, 6) => 50
            | _ => 500
            end
            in
            bindGen (freq [
              (_weight_1, returnGen 1%Z);
              (100-_weight_1, returnGen 0%Z)
            ]) (fun n1 =>
            (let _weight_2 := match (stack1, stack2) with
            | (4, 4) => 50
            | (4, 6) => 50
            | (6, 4) => 50
            | (6, 6) => 50
            | _ => 500
            end
            in
            bindGen (freq [
              (_weight_2, returnGen 2%Z);
              (100-_weight_2, returnGen 0%Z)
            ]) (fun n2 =>
            (let _weight_4 := match (stack1, stack2) with
            | (4, 4) => 50
            | (4, 6) => 50
            | (6, 4) => 50
            | (6, 6) => 50
            | _ => 500
            end
            in
            bindGen (freq [
              (_weight_4, returnGen 4%Z);
              (100-_weight_4, returnGen 0%Z)
            ]) (fun n4 =>
            (let _weight_8 := match (stack1, stack2) with
            | (4, 4) => 50
            | (4, 6) => 50
            | (6, 4) => 50
            | (6, 6) => 50
            | _ => 500
            end
            in
            bindGen (freq [
              (_weight_8, returnGen 8%Z);
              (100-_weight_8, returnGen 0%Z)
            ]) (fun n8 =>
            (let _weight_16 := match (stack1, stack2) with
            | (4, 4) => 50
            | (4, 6) => 50
            | (6, 4) => 50
            | (6, 6) => 50
            | _ => 500
            end
            in
            bindGen (freq [
              (_weight_16, returnGen 16%Z);
              (100-_weight_16, returnGen 0%Z)
            ]) (fun n16 =>
            (let _weight_32 := match (stack1, stack2) with
            | (4, 4) => 50
            | (4, 6) => 50
            | (6, 4) => 50
            | (6, 6) => 50
            | _ => 500
            end
            in
            bindGen (freq [
              (_weight_32, returnGen 32%Z);
              (100-_weight_32, returnGen 0%Z)
            ]) (fun n32 =>
              returnGen (n1 + n2 + n4 + n8 + n16 + n32)%Z
            )))))))))))) 
            (fun p3 => 
              (bindGen 
              (* GenZ3 *)
              (let _weight_1 := match (stack1, stack2) with
              | (4, 4) => 50
              | (4, 6) => 50
              | (6, 4) => 50
              | (6, 6) => 50
              | _ => 500
              end
              in
              bindGen (freq [
                (_weight_1, returnGen 1%Z);
                (100-_weight_1, returnGen 0%Z)
              ]) (fun n1 =>
              (let _weight_2 := match (stack1, stack2) with
              | (4, 4) => 50
              | (4, 6) => 50
              | (6, 4) => 50
              | (6, 6) => 50
              | _ => 500
              end
              in
              bindGen (freq [
                (_weight_2, returnGen 2%Z);
                (100-_weight_2, returnGen 0%Z)
              ]) (fun n2 =>
              (let _weight_4 := match (stack1, stack2) with
              | (4, 4) => 50
              | (4, 6) => 50
              | (6, 4) => 50
              | (6, 6) => 50
              | _ => 500
              end
              in
              bindGen (freq [
                (_weight_4, returnGen 4%Z);
                (100-_weight_4, returnGen 0%Z)
              ]) (fun n4 =>
              (let _weight_8 := match (stack1, stack2) with
              | (4, 4) => 50
              | (4, 6) => 50
              | (6, 4) => 50
              | (6, 6) => 50
              | _ => 500
              end
              in
              bindGen (freq [
                (_weight_8, returnGen 8%Z);
                (100-_weight_8, returnGen 0%Z)
              ]) (fun n8 =>
              (let _weight_16 := match (stack1, stack2) with
              | (4, 4) => 50
              | (4, 6) => 50
              | (6, 4) => 50
              | (6, 6) => 50
              | _ => 500
              end
              in
              bindGen (freq [
                (_weight_16, returnGen 16%Z);
                (100-_weight_16, returnGen 0%Z)
              ]) (fun n16 =>
              (let _weight_32 := match (stack1, stack2) with
              | (4, 4) => 50
              | (4, 6) => 50
              | (6, 4) => 50
              | (6, 6) => 50
              | _ => 500
              end
              in
              bindGen (freq [
                (_weight_32, returnGen 32%Z);
                (100-_weight_32, returnGen 0%Z)
              ]) (fun n32 =>
                returnGen (n1 + n2 + n4 + n8 + n16 + n32)%Z
              )))))))))))) 
              (fun p4 => 
                (bindGen (genLeafTree ctor3 stack2 5) 
                (fun p5 => 
                  (returnGen (T p1 p2 p3 p4 p5)))))))))))))))
    end
  | S size1 => match chosen_ctor with
    | CtorColorKVTree_E  => 
      (returnGen (E ))
    | CtorColorKVTree_T  => 
      (bindGen 
      (* Frequency3 *) (freq [
        (* 1 *) (match (size, stack1, stack2) with
        | (1, 4, 4) => 50
        | (1, 4, 6) => 50
        | (1, 6, 4) => 50
        | (1, 6, 6) => 50
        | (2, 4, 4) => 50
        | (2, 4, 6) => 50
        | (2, 6, 4) => 50
        | (2, 6, 6) => 50
        | (3, 4, 4) => 50
        | (3, 4, 6) => 50
        | (3, 6, 4) => 50
        | (3, 6, 6) => 50
        | (4, 4, 4) => 50
        | (4, 4, 6) => 50
        | (4, 6, 4) => 50
        | (4, 6, 6) => 50
        | (5, 4, 4) => 51
        | (5, 4, 6) => 52
        | (5, 6, 4) => 52
        | (5, 6, 6) => 50
        | (6, 0, 4) => 90
        | (6, 0, 6) => 90
        | (7, 0, 0) => 90
        | _ => 500
        end,
        (returnGen (MkLeafCtorColorCtorColorKVTreeCtorColorKVTree (LeafCtorColor_R ) (CtorColorKVTree_E ) (CtorColorKVTree_E )))); 
        (* 2 *) (match (size, stack1, stack2) with
        | (1, 4, 4) => 50
        | (1, 4, 6) => 50
        | (1, 6, 4) => 50
        | (1, 6, 6) => 50
        | (2, 4, 4) => 50
        | (2, 4, 6) => 50
        | (2, 6, 4) => 50
        | (2, 6, 6) => 50
        | (3, 4, 4) => 50
        | (3, 4, 6) => 50
        | (3, 6, 4) => 50
        | (3, 6, 6) => 50
        | (4, 4, 4) => 50
        | (4, 4, 6) => 50
        | (4, 6, 4) => 50
        | (4, 6, 6) => 50
        | (5, 4, 4) => 50
        | (5, 4, 6) => 50
        | (5, 6, 4) => 50
        | (5, 6, 6) => 50
        | (6, 0, 4) => 10
        | (6, 0, 6) => 10
        | (7, 0, 0) => 90
        | _ => 500
        end,
        (returnGen (MkLeafCtorColorCtorColorKVTreeCtorColorKVTree (LeafCtorColor_B ) (CtorColorKVTree_E ) (CtorColorKVTree_E )))); 
        (* 3 *) (match (size, stack1, stack2) with
        | (1, 4, 4) => 50
        | (1, 4, 6) => 50
        | (1, 6, 4) => 50
        | (1, 6, 6) => 50
        | (2, 4, 4) => 50
        | (2, 4, 6) => 50
        | (2, 6, 4) => 50
        | (2, 6, 6) => 50
        | (3, 4, 4) => 50
        | (3, 4, 6) => 50
        | (3, 6, 4) => 50
        | (3, 6, 6) => 50
        | (4, 4, 4) => 50
        | (4, 4, 6) => 50
        | (4, 6, 4) => 50
        | (4, 6, 6) => 50
        | (5, 4, 4) => 50
        | (5, 4, 6) => 50
        | (5, 6, 4) => 50
        | (5, 6, 6) => 50
        | (6, 0, 4) => 10
        | (6, 0, 6) => 10
        | (7, 0, 0) => 10
        | _ => 500
        end,
        (returnGen (MkLeafCtorColorCtorColorKVTreeCtorColorKVTree (LeafCtorColor_R ) (CtorColorKVTree_T ) (CtorColorKVTree_E )))); 
        (* 4 *) (match (size, stack1, stack2) with
        | (1, 4, 4) => 50
        | (1, 4, 6) => 50
        | (1, 6, 4) => 50
        | (1, 6, 6) => 50
        | (2, 4, 4) => 50
        | (2, 4, 6) => 50
        | (2, 6, 4) => 50
        | (2, 6, 6) => 50
        | (3, 4, 4) => 50
        | (3, 4, 6) => 50
        | (3, 6, 4) => 50
        | (3, 6, 6) => 50
        | (4, 4, 4) => 50
        | (4, 4, 6) => 50
        | (4, 6, 4) => 50
        | (4, 6, 6) => 50
        | (5, 4, 4) => 50
        | (5, 4, 6) => 50
        | (5, 6, 4) => 50
        | (5, 6, 6) => 50
        | (6, 0, 4) => 10
        | (6, 0, 6) => 10
        | (7, 0, 0) => 90
        | _ => 500
        end,
        (returnGen (MkLeafCtorColorCtorColorKVTreeCtorColorKVTree (LeafCtorColor_B ) (CtorColorKVTree_T ) (CtorColorKVTree_E )))); 
        (* 5 *) (match (size, stack1, stack2) with
        | (1, 4, 4) => 50
        | (1, 4, 6) => 50
        | (1, 6, 4) => 50
        | (1, 6, 6) => 50
        | (2, 4, 4) => 50
        | (2, 4, 6) => 50
        | (2, 6, 4) => 50
        | (2, 6, 6) => 50
        | (3, 4, 4) => 50
        | (3, 4, 6) => 50
        | (3, 6, 4) => 50
        | (3, 6, 6) => 50
        | (4, 4, 4) => 50
        | (4, 4, 6) => 50
        | (4, 6, 4) => 50
        | (4, 6, 6) => 50
        | (5, 4, 4) => 50
        | (5, 4, 6) => 50
        | (5, 6, 4) => 50
        | (5, 6, 6) => 50
        | (6, 0, 4) => 10
        | (6, 0, 6) => 10
        | (7, 0, 0) => 10
        | _ => 500
        end,
        (returnGen (MkLeafCtorColorCtorColorKVTreeCtorColorKVTree (LeafCtorColor_R ) (CtorColorKVTree_E ) (CtorColorKVTree_T )))); 
        (* 6 *) (match (size, stack1, stack2) with
        | (1, 4, 4) => 50
        | (1, 4, 6) => 50
        | (1, 6, 4) => 50
        | (1, 6, 6) => 50
        | (2, 4, 4) => 50
        | (2, 4, 6) => 50
        | (2, 6, 4) => 50
        | (2, 6, 6) => 50
        | (3, 4, 4) => 50
        | (3, 4, 6) => 50
        | (3, 6, 4) => 50
        | (3, 6, 6) => 50
        | (4, 4, 4) => 50
        | (4, 4, 6) => 50
        | (4, 6, 4) => 50
        | (4, 6, 6) => 50
        | (5, 4, 4) => 50
        | (5, 4, 6) => 50
        | (5, 6, 4) => 50
        | (5, 6, 6) => 50
        | (6, 0, 4) => 10
        | (6, 0, 6) => 10
        | (7, 0, 0) => 39
        | _ => 500
        end,
        (returnGen (MkLeafCtorColorCtorColorKVTreeCtorColorKVTree (LeafCtorColor_B ) (CtorColorKVTree_E ) (CtorColorKVTree_T )))); 
        (* 7 *) (match (size, stack1, stack2) with
        | (1, 4, 4) => 50
        | (1, 4, 6) => 50
        | (1, 6, 4) => 50
        | (1, 6, 6) => 50
        | (2, 4, 4) => 50
        | (2, 4, 6) => 50
        | (2, 6, 4) => 50
        | (2, 6, 6) => 50
        | (3, 4, 4) => 50
        | (3, 4, 6) => 50
        | (3, 6, 4) => 50
        | (3, 6, 6) => 50
        | (4, 4, 4) => 50
        | (4, 4, 6) => 50
        | (4, 6, 4) => 50
        | (4, 6, 6) => 50
        | (5, 4, 4) => 50
        | (5, 4, 6) => 50
        | (5, 6, 4) => 50
        | (5, 6, 6) => 50
        | (6, 0, 4) => 10
        | (6, 0, 6) => 10
        | (7, 0, 0) => 10
        | _ => 500
        end,
        (returnGen (MkLeafCtorColorCtorColorKVTreeCtorColorKVTree (LeafCtorColor_R ) (CtorColorKVTree_T ) (CtorColorKVTree_T )))); 
        (* 8 *) (match (size, stack1, stack2) with
        | (1, 4, 4) => 50
        | (1, 4, 6) => 50
        | (1, 6, 4) => 50
        | (1, 6, 6) => 50
        | (2, 4, 4) => 50
        | (2, 4, 6) => 50
        | (2, 6, 4) => 50
        | (2, 6, 6) => 50
        | (3, 4, 4) => 50
        | (3, 4, 6) => 50
        | (3, 6, 4) => 50
        | (3, 6, 6) => 50
        | (4, 4, 4) => 50
        | (4, 4, 6) => 50
        | (4, 6, 4) => 50
        | (4, 6, 6) => 50
        | (5, 4, 4) => 50
        | (5, 4, 6) => 50
        | (5, 6, 4) => 50
        | (5, 6, 6) => 50
        | (6, 0, 4) => 10
        | (6, 0, 6) => 10
        | (7, 0, 0) => 10
        | _ => 500
        end,
        (returnGen (MkLeafCtorColorCtorColorKVTreeCtorColorKVTree (LeafCtorColor_B ) (CtorColorKVTree_T ) (CtorColorKVTree_T ))))]) 
      (fun param_variantis => (let '(MkLeafCtorColorCtorColorKVTreeCtorColorKVTree ctor1 ctor2 ctor3) := param_variantis in

        (bindGen (genLeafColor ctor1 stack2 2) 
        (fun p1 => 
          (bindGen (genTree size1 ctor2 stack2 4) 
          (fun p2 => 
            (bindGen 
            (* GenZ2 *)
            (let _weight_1 := match (size, stack1, stack2) with
            | (1, 4, 4) => 50
            | (1, 4, 6) => 50
            | (1, 6, 4) => 50
            | (1, 6, 6) => 50
            | (2, 4, 4) => 50
            | (2, 4, 6) => 50
            | (2, 6, 4) => 50
            | (2, 6, 6) => 50
            | (3, 4, 4) => 50
            | (3, 4, 6) => 50
            | (3, 6, 4) => 50
            | (3, 6, 6) => 50
            | (4, 4, 4) => 50
            | (4, 4, 6) => 50
            | (4, 6, 4) => 50
            | (4, 6, 6) => 50
            | (5, 4, 4) => 48
            | (5, 4, 6) => 50
            | (5, 6, 4) => 47
            | (5, 6, 6) => 50
            | (6, 0, 4) => 32
            | (6, 0, 6) => 65
            | (7, 0, 0) => 67
            | _ => 500
            end
            in
            bindGen (freq [
              (_weight_1, returnGen 1%Z);
              (100-_weight_1, returnGen 0%Z)
            ]) (fun n1 =>
            (let _weight_2 := match (size, stack1, stack2) with
            | (1, 4, 4) => 50
            | (1, 4, 6) => 50
            | (1, 6, 4) => 50
            | (1, 6, 6) => 50
            | (2, 4, 4) => 50
            | (2, 4, 6) => 50
            | (2, 6, 4) => 50
            | (2, 6, 6) => 50
            | (3, 4, 4) => 50
            | (3, 4, 6) => 50
            | (3, 6, 4) => 50
            | (3, 6, 6) => 50
            | (4, 4, 4) => 50
            | (4, 4, 6) => 50
            | (4, 6, 4) => 50
            | (4, 6, 6) => 50
            | (5, 4, 4) => 48
            | (5, 4, 6) => 47
            | (5, 6, 4) => 47
            | (5, 6, 6) => 50
            | (6, 0, 4) => 65
            | (6, 0, 6) => 54
            | (7, 0, 0) => 43
            | _ => 500
            end
            in
            bindGen (freq [
              (_weight_2, returnGen 2%Z);
              (100-_weight_2, returnGen 0%Z)
            ]) (fun n2 =>
            (let _weight_4 := match (size, stack1, stack2) with
            | (1, 4, 4) => 50
            | (1, 4, 6) => 50
            | (1, 6, 4) => 50
            | (1, 6, 6) => 50
            | (2, 4, 4) => 50
            | (2, 4, 6) => 50
            | (2, 6, 4) => 50
            | (2, 6, 6) => 50
            | (3, 4, 4) => 50
            | (3, 4, 6) => 50
            | (3, 6, 4) => 50
            | (3, 6, 6) => 50
            | (4, 4, 4) => 50
            | (4, 4, 6) => 50
            | (4, 6, 4) => 50
            | (4, 6, 6) => 50
            | (5, 4, 4) => 48
            | (5, 4, 6) => 53
            | (5, 6, 4) => 47
            | (5, 6, 6) => 50
            | (6, 0, 4) => 45
            | (6, 0, 6) => 86
            | (7, 0, 0) => 41
            | _ => 500
            end
            in
            bindGen (freq [
              (_weight_4, returnGen 4%Z);
              (100-_weight_4, returnGen 0%Z)
            ]) (fun n4 =>
            (let _weight_8 := match (size, stack1, stack2) with
            | (1, 4, 4) => 50
            | (1, 4, 6) => 50
            | (1, 6, 4) => 50
            | (1, 6, 6) => 50
            | (2, 4, 4) => 50
            | (2, 4, 6) => 50
            | (2, 6, 4) => 50
            | (2, 6, 6) => 50
            | (3, 4, 4) => 50
            | (3, 4, 6) => 50
            | (3, 6, 4) => 50
            | (3, 6, 6) => 50
            | (4, 4, 4) => 50
            | (4, 4, 6) => 50
            | (4, 6, 4) => 50
            | (4, 6, 6) => 50
            | (5, 4, 4) => 48
            | (5, 4, 6) => 47
            | (5, 6, 4) => 47
            | (5, 6, 6) => 50
            | (6, 0, 4) => 17
            | (6, 0, 6) => 84
            | (7, 0, 0) => 46
            | _ => 500
            end
            in
            bindGen (freq [
              (_weight_8, returnGen 8%Z);
              (100-_weight_8, returnGen 0%Z)
            ]) (fun n8 =>
            (let _weight_16 := match (size, stack1, stack2) with
            | (1, 4, 4) => 50
            | (1, 4, 6) => 50
            | (1, 6, 4) => 50
            | (1, 6, 6) => 50
            | (2, 4, 4) => 50
            | (2, 4, 6) => 50
            | (2, 6, 4) => 50
            | (2, 6, 6) => 50
            | (3, 4, 4) => 50
            | (3, 4, 6) => 50
            | (3, 6, 4) => 50
            | (3, 6, 6) => 50
            | (4, 4, 4) => 50
            | (4, 4, 6) => 50
            | (4, 6, 4) => 50
            | (4, 6, 6) => 50
            | (5, 4, 4) => 48
            | (5, 4, 6) => 47
            | (5, 6, 4) => 50
            | (5, 6, 6) => 50
            | (6, 0, 4) => 12
            | (6, 0, 6) => 90
            | (7, 0, 0) => 43
            | _ => 500
            end
            in
            bindGen (freq [
              (_weight_16, returnGen 16%Z);
              (100-_weight_16, returnGen 0%Z)
            ]) (fun n16 =>
            (let _weight_32 := match (size, stack1, stack2) with
            | (1, 4, 4) => 50
            | (1, 4, 6) => 50
            | (1, 6, 4) => 50
            | (1, 6, 6) => 50
            | (2, 4, 4) => 50
            | (2, 4, 6) => 50
            | (2, 6, 4) => 50
            | (2, 6, 6) => 50
            | (3, 4, 4) => 50
            | (3, 4, 6) => 50
            | (3, 6, 4) => 50
            | (3, 6, 6) => 50
            | (4, 4, 4) => 50
            | (4, 4, 6) => 50
            | (4, 6, 4) => 50
            | (4, 6, 6) => 50
            | (5, 4, 4) => 48
            | (5, 4, 6) => 50
            | (5, 6, 4) => 50
            | (5, 6, 6) => 50
            | (6, 0, 4) => 10
            | (6, 0, 6) => 90
            | (7, 0, 0) => 78
            | _ => 500
            end
            in
            bindGen (freq [
              (_weight_32, returnGen 32%Z);
              (100-_weight_32, returnGen 0%Z)
            ]) (fun n32 =>
              returnGen (n1 + n2 + n4 + n8 + n16 + n32)%Z
            )))))))))))) 
            (fun p3 => 
              (bindGen 
              (* GenZ4 *)
              (let _weight_1 := match (size, stack1, stack2) with
              | (1, 4, 4) => 50
              | (1, 4, 6) => 50
              | (1, 6, 4) => 50
              | (1, 6, 6) => 50
              | (2, 4, 4) => 50
              | (2, 4, 6) => 50
              | (2, 6, 4) => 50
              | (2, 6, 6) => 50
              | (3, 4, 4) => 50
              | (3, 4, 6) => 50
              | (3, 6, 4) => 50
              | (3, 6, 6) => 50
              | (4, 4, 4) => 50
              | (4, 4, 6) => 50
              | (4, 6, 4) => 50
              | (4, 6, 6) => 50
              | (5, 4, 4) => 48
              | (5, 4, 6) => 53
              | (5, 6, 4) => 53
              | (5, 6, 6) => 50
              | (6, 0, 4) => 78
              | (6, 0, 6) => 45
              | (7, 0, 0) => 62
              | _ => 500
              end
              in
              bindGen (freq [
                (_weight_1, returnGen 1%Z);
                (100-_weight_1, returnGen 0%Z)
              ]) (fun n1 =>
              (let _weight_2 := match (size, stack1, stack2) with
              | (1, 4, 4) => 50
              | (1, 4, 6) => 50
              | (1, 6, 4) => 50
              | (1, 6, 6) => 50
              | (2, 4, 4) => 50
              | (2, 4, 6) => 50
              | (2, 6, 4) => 50
              | (2, 6, 6) => 50
              | (3, 4, 4) => 50
              | (3, 4, 6) => 50
              | (3, 6, 4) => 50
              | (3, 6, 6) => 50
              | (4, 4, 4) => 50
              | (4, 4, 6) => 50
              | (4, 6, 4) => 50
              | (4, 6, 6) => 50
              | (5, 4, 4) => 48
              | (5, 4, 6) => 50
              | (5, 6, 4) => 50
              | (5, 6, 6) => 50
              | (6, 0, 4) => 37
              | (6, 0, 6) => 42
              | (7, 0, 0) => 47
              | _ => 500
              end
              in
              bindGen (freq [
                (_weight_2, returnGen 2%Z);
                (100-_weight_2, returnGen 0%Z)
              ]) (fun n2 =>
              (let _weight_4 := match (size, stack1, stack2) with
              | (1, 4, 4) => 50
              | (1, 4, 6) => 50
              | (1, 6, 4) => 50
              | (1, 6, 6) => 50
              | (2, 4, 4) => 50
              | (2, 4, 6) => 50
              | (2, 6, 4) => 50
              | (2, 6, 6) => 50
              | (3, 4, 4) => 50
              | (3, 4, 6) => 50
              | (3, 6, 4) => 50
              | (3, 6, 6) => 50
              | (4, 4, 4) => 50
              | (4, 4, 6) => 50
              | (4, 6, 4) => 50
              | (4, 6, 6) => 50
              | (5, 4, 4) => 52
              | (5, 4, 6) => 47
              | (5, 6, 4) => 50
              | (5, 6, 6) => 50
              | (6, 0, 4) => 33
              | (6, 0, 6) => 73
              | (7, 0, 0) => 69
              | _ => 500
              end
              in
              bindGen (freq [
                (_weight_4, returnGen 4%Z);
                (100-_weight_4, returnGen 0%Z)
              ]) (fun n4 =>
              (let _weight_8 := match (size, stack1, stack2) with
              | (1, 4, 4) => 50
              | (1, 4, 6) => 50
              | (1, 6, 4) => 50
              | (1, 6, 6) => 50
              | (2, 4, 4) => 50
              | (2, 4, 6) => 50
              | (2, 6, 4) => 50
              | (2, 6, 6) => 50
              | (3, 4, 4) => 50
              | (3, 4, 6) => 50
              | (3, 6, 4) => 50
              | (3, 6, 6) => 50
              | (4, 4, 4) => 50
              | (4, 4, 6) => 50
              | (4, 6, 4) => 50
              | (4, 6, 6) => 50
              | (5, 4, 4) => 48
              | (5, 4, 6) => 47
              | (5, 6, 4) => 50
              | (5, 6, 6) => 50
              | (6, 0, 4) => 56
              | (6, 0, 6) => 37
              | (7, 0, 0) => 41
              | _ => 500
              end
              in
              bindGen (freq [
                (_weight_8, returnGen 8%Z);
                (100-_weight_8, returnGen 0%Z)
              ]) (fun n8 =>
              (let _weight_16 := match (size, stack1, stack2) with
              | (1, 4, 4) => 50
              | (1, 4, 6) => 50
              | (1, 6, 4) => 50
              | (1, 6, 6) => 50
              | (2, 4, 4) => 50
              | (2, 4, 6) => 50
              | (2, 6, 4) => 50
              | (2, 6, 6) => 50
              | (3, 4, 4) => 50
              | (3, 4, 6) => 50
              | (3, 6, 4) => 50
              | (3, 6, 6) => 50
              | (4, 4, 4) => 50
              | (4, 4, 6) => 50
              | (4, 6, 4) => 50
              | (4, 6, 6) => 50
              | (5, 4, 4) => 52
              | (5, 4, 6) => 50
              | (5, 6, 4) => 47
              | (5, 6, 6) => 50
              | (6, 0, 4) => 45
              | (6, 0, 6) => 24
              | (7, 0, 0) => 52
              | _ => 500
              end
              in
              bindGen (freq [
                (_weight_16, returnGen 16%Z);
                (100-_weight_16, returnGen 0%Z)
              ]) (fun n16 =>
              (let _weight_32 := match (size, stack1, stack2) with
              | (1, 4, 4) => 50
              | (1, 4, 6) => 50
              | (1, 6, 4) => 50
              | (1, 6, 6) => 50
              | (2, 4, 4) => 50
              | (2, 4, 6) => 50
              | (2, 6, 4) => 50
              | (2, 6, 6) => 50
              | (3, 4, 4) => 50
              | (3, 4, 6) => 50
              | (3, 6, 4) => 50
              | (3, 6, 6) => 50
              | (4, 4, 4) => 50
              | (4, 4, 6) => 50
              | (4, 6, 4) => 50
              | (4, 6, 6) => 50
              | (5, 4, 4) => 48
              | (5, 4, 6) => 53
              | (5, 6, 4) => 47
              | (5, 6, 6) => 50
              | (6, 0, 4) => 69
              | (6, 0, 6) => 22
              | (7, 0, 0) => 34
              | _ => 500
              end
              in
              bindGen (freq [
                (_weight_32, returnGen 32%Z);
                (100-_weight_32, returnGen 0%Z)
              ]) (fun n32 =>
                returnGen (n1 + n2 + n4 + n8 + n16 + n32)%Z
              )))))))))))) 
              (fun p4 => 
                (bindGen (genTree size1 ctor3 stack2 6) 
                (fun p5 => 
                  (returnGen (T p1 p2 p3 p4 p5)))))))))))))))
    end
  end.

Definition gSized :=

  (bindGen 
  (* Frequency1 *) (freq [
    (* 1 *) (match (tt) with
    | tt => 10
    end,
    (returnGen (CtorColorKVTree_E ))); 
    (* 2 *) (match (tt) with
    | tt => 90
    end,
    (returnGen (CtorColorKVTree_T )))]) 
  (fun init_ctor => (genTree 7 init_ctor 0 0))).

(* --------------------- Tests --------------------- *)

Definition test_prop_InsertValid :=  
    forAll gSized (fun t =>    
    forAll arbitrary (fun k =>
    forAll arbitrary (fun v =>
        (prop_InsertValid t k v)))).

(*! QuickChick test_prop_InsertValid. *)

Definition test_prop_DeleteValid :=  
    forAll gSized (fun t =>    
    forAll arbitrary (fun k =>
        prop_DeleteValid t k)).

(*! QuickChick test_prop_DeleteValid. *)

Definition test_prop_InsertPost :=  
    forAll gSized (fun t =>    
    forAll arbitrary (fun k =>
    forAll arbitrary (fun k' =>
     forAll arbitrary (fun v =>
        prop_InsertPost t k k' v)))).

(*! QuickChick test_prop_InsertPost. *)

Definition test_prop_DeletePost := 
    forAll gSized (fun t =>    
    forAll arbitrary (fun k =>
    forAll arbitrary (fun k' =>
        prop_DeletePost t k k'))).

(*! QuickChick test_prop_DeletePost. *)
    
Definition test_prop_InsertModel :=  
    forAll gSized (fun t =>    
    forAll arbitrary (fun k =>
    forAll arbitrary (fun v =>
        prop_InsertModel t k v))).

(*! QuickChick test_prop_InsertModel. *)
    
Definition test_prop_DeleteModel :=  
    forAll gSized (fun t =>    
    forAll arbitrary (fun k =>
            prop_DeleteModel t k)).

(*! QuickChick test_prop_DeleteModel. *)

Definition test_prop_InsertInsert :=  
    forAll gSized (fun t =>    
    forAll arbitrary (fun k =>
    forAll arbitrary (fun k' =>
    forAll arbitrary (fun v =>
    forAll arbitrary (fun v' =>     
        prop_InsertInsert t k k' v v'))))).

(*! QuickChick test_prop_InsertInsert. *)
    
Definition test_prop_InsertDelete := 
    forAll gSized (fun t =>    
    forAll arbitrary (fun k =>
    forAll arbitrary (fun k' =>
    forAll arbitrary (fun v =>
        prop_InsertDelete t k k' v)))).

(*! QuickChick test_prop_InsertDelete. *)
    
Definition test_prop_DeleteInsert := 
    forAll gSized (fun t =>    
    forAll arbitrary (fun k =>
    forAll arbitrary (fun k' =>
    forAll arbitrary (fun v' =>
        prop_DeleteInsert t k k' v')))).

(*! QuickChick test_prop_DeleteInsert. *)
    
Definition test_prop_DeleteDelete :=  
    forAll gSized (fun t =>    
    forAll arbitrary (fun k =>
    forAll arbitrary (fun k' =>
        prop_DeleteDelete t k k'))).

(*! QuickChick test_prop_DeleteDelete. *)
          
